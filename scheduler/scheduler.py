import queue
import time
import math
from concurrent.futures import Future

import torch
from typing import ClassVar
from attn.core import use_triton_attention
from kv_cache.storage import KVStorage
from scheduler.bucket import Bucket


class Scheduler:

    DECODE_BATCH_BUCKETS: ClassVar[tuple] = (1, 2, 4, 8, 16)
    KV_CACHE_CAPACITY: ClassVar[int] = 32768

    SENTINEL: ClassVar[object] = object()

    def __init__(
            self,
            model,
            eos_token_ids: set[int],
            max_new_tokens: int,
            n_slots: int,
            page_size: int,
            dtype: torch.dtype
    ) -> None:

        if n_slots <= 1:
            raise ValueError(
                "n_slots must be greater than 1."
            )

        if n_slots - 1 > self.DECODE_BATCH_BUCKETS[-1]:
            raise ValueError(
                "n_slots can't be larger than largest bucket size."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive."
            )

        if not eos_token_ids:
            raise ValueError(
                "eos_token_ids must not be empty."
            )

        try:
            use_triton_attention(model)

        except Exception as exc:
            raise RuntimeError(
                "Failed to switch triton attention:"
                f" {exc}"
            )

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.n_slots = n_slots
        self.page_size = page_size
        self.eos_token_ids = frozenset(eos_token_ids)

        self.kv_cache = KVStorage(
            config=model.config,
            max_cache_len=self.KV_CACHE_CAPACITY,
            n_slots=n_slots,
            page_size=page_size,
            device=model.device,
            dtype=dtype
        )
        self.n_pages = self.kv_cache.n_pages

        if self._required_pages(1) > self.n_pages - 1:
            raise ValueError(
                f"max_new_tokens can't need more than {self.n_pages - 1} pages."
            )

        self.buckets = self._generate_buckets()
        self.requests = queue.Queue()

        self.futs: list[Future | None] = []
        self.new_tokens: list[list[int]] = []
        self.allocations: list[set] = []
        self.lens: list[int] = []
        self._init(self.n_slots)

        self.t_arrivals: dict[int, float] = {}
        self.t_firsts: dict[int, float] = {}

        self.page_ids = set(range(self.n_pages))
        self.slot_ids = set(range(self.n_slots))
        self.open = True

    def _bucket_size(self, batch_size: int) -> int:
        for size in self.DECODE_BATCH_BUCKETS:
            if size >= batch_size:
                return size

        raise RuntimeError(
            "Invalid batch size."
        )

    def _generate_buckets(self) -> dict[int, Bucket]:
        buckets: dict[int, Bucket] = {}
        for size in self.DECODE_BATCH_BUCKETS:
            buckets[size] = Bucket(
                model=self.model,
                kv_cache=self.kv_cache,
                batch_size=size
            )

            if size >= self.n_slots - 1:
                break

        return buckets

    def _get_bucket(self, batch_size: int) -> Bucket:
        bucket_size = self._bucket_size(batch_size)
        return self.buckets[bucket_size]

    @property
    def _empty_pages(self) -> set[int]:
        occupied_pages = set()

        for pages in self.allocations:
            occupied_pages.update(pages)

        return self.page_ids - occupied_pages

    @property
    def _filled_pages(self) -> set[int]:
        occupied_pages = set()

        for pages in self.allocations:
            occupied_pages.update(pages)

        return occupied_pages - {self.kv_cache.unused_idx}

    @property
    def _empty_slots(self) -> list[int]:
        return [
            i
            for i, fut in enumerate(self.futs)
            if i != self.kv_cache.unused_idx and fut is None
        ]

    @property
    def _filled_slots(self) -> list[int]:
        return [
            i
            for i, fut in enumerate(self.futs)
            if i != self.kv_cache.unused_idx and fut is not None
        ]

    def _required_pages(self, prompt_len: int) -> int:
        return math.ceil((prompt_len + self.max_new_tokens) / self.page_size)

    def _validate_palloc(
            self,
            slot_ids: list[int],
            page_counts: list[int]
    ) -> list[int]:

        if len(page_counts) != len(slot_ids):
            raise RuntimeError(
                "Unmatching page counts."
            )

        if any(id_ < 0 or id_ == self.kv_cache.unused_idx or id_ >= self.n_slots for id_ in slot_ids):
            raise RuntimeError(
                "Invalid slot ids."
            )

        if any(self.lens[id_] % self.page_size != 0 for id_ in slot_ids):
            raise RuntimeError(
                "Cannot allocate pages to slots with unfilled last page."
            )

        if any(
                count <= 0
                or count > self.n_pages - 1 - self.lens[id_] // self.page_size
                for id_, count in zip(slot_ids, page_counts)
        ):
            raise RuntimeError(
                "Invalid page counts."
            )

        empty_pages = list(self._empty_pages)
        if sum(page_counts) > len(empty_pages):
            raise RuntimeError(
                "No enough pages to allocate."
            )

        return empty_pages

    def _palloc(
            self,
            slot_ids: list[int],
            page_counts: list[int]
    ) -> None:

        empty_pages = self._validate_palloc(
            slot_ids,
            page_counts
        )

        cursor = 0
        for id_, page_count in zip(slot_ids, page_counts):
            next_ = self.lens[id_] // self.page_size
            piece = empty_pages[cursor: cursor + page_count]

            self.kv_cache.block_table[
                id_, next_:next_ + page_count
            ] = piece

            self.allocations[id_].update(piece)

            cursor += page_count

    def _validate_pfree(
            self,
            slot_ids: list[int]
    ) -> None:

        if any(id_ < 0 or id_ == self.kv_cache.unused_idx or id_ >= self.n_slots for id_ in slot_ids):
            raise RuntimeError(
                "Invalid slot ids."
            )

        if any(not self.allocations[id_] for id_ in slot_ids):
            raise RuntimeError(
                "Cannot free a slot with no pages."
            )

    def _pfree(
            self,
            slot_ids: list[int]
    ) -> None:
        self._validate_pfree(slot_ids)
        self._clear(slot_ids)

    def _refresh_slots(
            self,
            slot_ids: list[int],
            next_tokens: list[int]
    ) -> None:
        finished_slots = []
        current = time.perf_counter()

        for idx, token in enumerate(next_tokens):
            slot_id = slot_ids[idx]

            if (
                    token in self.eos_token_ids
                    or len(self.new_tokens[slot_id]) == self.max_new_tokens - 1
            ):
                t_arrival = self.t_arrivals.get(slot_id)
                t_first = self.t_firsts.get(slot_id, current)
                new_tokens = self.new_tokens[slot_id] + [token]

                self.futs[slot_id].set_result(
                    (new_tokens, t_arrival, t_first, current)
                )

                finished_slots.append(slot_id)

            else:
                if not self.new_tokens[slot_id]:
                    self.t_firsts[slot_id] = current
                    self.new_tokens[slot_id] = [token]

                else:
                    self.new_tokens[slot_id].append(token)

        self._pfree(finished_slots)

    def _loop_prefill(self, wait: bool) -> None:
        prefill_prompts: list[list[int]] = []
        prefill_slot_ids: list[int] = []

        for id_ in self._empty_slots():
            if wait:
                req = self.requests.get()
                wait = False
            else:
                try:
                    req = self.requests.get_nowait()
                except queue.Empty:
                    break

            if req is self.SENTINEL:
                self.open = False
                break

            prompt, fut, t_arrival = req
            self.futs[id_] = fut
            self.t_arrivals[id_] = t_arrival

            prefill_prompts.append(prompt)
            prefill_slot_ids.append(id_)

        if prefill_prompts:
            bucket = self._get_bucket(
                batch_size=len(prefill_prompts)
            )
            next_tokens = bucket.prefill_module.forward(
                prompts=prefill_prompts,
                slot_ids=prefill_slot_ids
            )

            self._refresh_slots(
                prefill_slot_ids, next_tokens
            )

    def _loop_decode(self) -> None:
        slot_ids = self._filled_slots()

        bucket = self._get_bucket(
            batch_size=len(slot_ids)
        )
        next_tokens = bucket.decode_module.forward(
            slot_ids=slot_ids
        )

        self._refresh_slots(
            slot_ids, next_tokens
        )

    def _consume_queue(self, exc: Exception) -> None:
        while True:
            try:
                req = self.requests.get_nowait()
            except queue.Empty:
                break

            if req is self.SENTINEL:
                self.open = False
                continue

            _, fut, _ = req
            fut.set_exception(exc)

    def _clear(self, ids: list[int] | None = None) -> None:

        if ids is None:
            self.futs.clear()
            self.new_tokens.clear()
            self.t_arrivals.clear()
            self.t_firsts.clear()
            self.allocations.clear()
            self.lens.clear()
            self.kv_cache.reset(list(self.slot_ids))
            self._init(self.n_slots)

        else:
            for id_ in ids:
                self.futs[id_] = None
                self.new_tokens[id_] = []
                self.t_arrivals.pop(id_, None)
                self.t_firsts.pop(id_, None)
                self.allocations[id_] = set()
                self.lens[id_] = 0
            self.kv_cache.reset(ids)

    def _init(self, n_slots: int) -> None:
        for i in range(n_slots):
            self.futs.append(None)
            self.new_tokens.append([])
            self.allocations.append(
                set()
                if i != self.kv_cache.unused_idx
                else {self.kv_cache.unused_idx}
            )
            self.lens.append(0)

    def call(self, batch: list[list[int]]) -> list[Future]:
        futs = []
        t_arrival = time.perf_counter()

        for prompt in batch:
            fut = Future()

            l = len(prompt)
            if not self.open:
                fut.set_exception(RuntimeError(
                    "Loop is not open."
                ))

            elif l == 0:
                fut.set_exception(ValueError(
                    "Empty prompts are not allowed."
                ))

            elif self._required_pages(l) > self.n_pages - 1:
                fut.set_exception(ValueError(
                    "Possible amount of tokens exceeds available pages."
                ))

            else:
                self.requests.put((prompt, fut, t_arrival))

            futs.append(fut)

        return futs

    def stop(self) -> None:
        self.requests.put(self.SENTINEL)

    def loop(self) -> None:

        while True:
            try:
                if self.open:
                    if len(self._filled_slots()) > 0:
                        self._loop_prefill(False)

                    else:
                        self._loop_prefill(True)

                if self._filled_slots():
                    self._loop_decode()

                elif not self.open:
                    self._consume_queue(exc=RuntimeError(
                        "Loop has been terminated."
                    ))
                    self._clear()
                    break

            except Exception as exc:
                for id_ in self._filled_slots():
                    self.futs[id_].set_exception(exc)

                self._consume_queue(exc=exc)
                self._clear()
