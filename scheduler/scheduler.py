import queue
import time
import math
from concurrent.futures import Future

import torch
from typing import ClassVar
from attn.core import use_triton_attention
from scheduler.models import Request
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

        if self._required_pages(1, True) > self.n_pages - 1:
            raise ValueError(
                f"max_new_tokens can't need more than {self.n_pages - 1} pages."
            )

        self.buckets = self._generate_buckets()

        self.requests = queue.Queue()
        self.preempted: list[Request] = []

        self.slot_listing_order: list[int] = []
        self.slots: list[Request | None] = [
            None for _ in range(self.n_slots)
        ]

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
        occupied_pages = {self.kv_cache.unused_idx}

        for req in self.slots:
            if req is not None:
                occupied_pages.update(req.pages)

        return self.page_ids - occupied_pages

    @property
    def _filled_pages(self) -> set[int]:
        occupied_pages = set()

        for req in self.slots:
            if req is not None:
                occupied_pages.update(req.pages)

        return occupied_pages - {self.kv_cache.unused_idx}

    @property
    def _empty_slots(self) -> list[int]:
        return [
            i
            for i, req in enumerate(self.slots)
            if i != self.kv_cache.unused_idx and req is None
        ]

    @property
    def _filled_slots(self) -> list[int]:
        return [
            i
            for i, req in enumerate(self.slots)
            if i != self.kv_cache.unused_idx and req is not None
        ]

    def _required_pages(
            self,
            prompt_len: int,
            total: bool
    ) -> int:
        L = prompt_len + (self.max_new_tokens if total else 0)
        if L % self.page_size == 0:
            L += 1

        return math.ceil(L / self.page_size)

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

        if any(
                not self.slots[id_].requires_page(self.page_size)
                for id_ in slot_ids
        ):
            raise RuntimeError(
                "Cannot allocate pages to slots with unfilled last page."
            )

        if any(
                count <= 0
                or count > self.n_pages - 1 - len(self.slots[id_].pages)
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
            req = self.slots[id_]

            next_ = len(req.pages)
            piece = empty_pages[cursor: cursor + page_count]

            self.kv_cache.block_table[
                id_, next_:next_ + page_count
            ] = torch.tensor(piece, device=self.kv_cache.device)

            req.pages.update(piece)

            cursor += page_count

    def _validate_pfree(
            self,
            slot_ids: list[int]
    ) -> None:

        if any(id_ < 0 or id_ == self.kv_cache.unused_idx or id_ >= self.n_slots for id_ in slot_ids):
            raise RuntimeError(
                "Invalid slot ids."
            )

        if any(not self.slots[id_].pages for id_ in slot_ids):
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
            req = self.slots[slot_id]

            if not req.prefilled:
                req.kv_len = (len(req.prompt)
                              + len(req.new_tokens))
                req.prefilled = True

            else:
                req.kv_len += 1

            if (
                    token in self.eos_token_ids
                    or len(req.new_tokens) == self.max_new_tokens - 1
            ):
                t_arrival = req.t_arrival
                t_first = req.t_first or current
                new_tokens = req.new_tokens + [token]

                req.future.set_result(
                    (new_tokens, t_arrival, t_first, current)
                )

                finished_slots.append(slot_id)

            else:
                if not req.new_tokens:
                    req.t_first = current
                    req.new_tokens = [token]

                else:
                    req.new_tokens.append(token)


        self._pfree(finished_slots)

    def _loop_prefill(self, wait: bool, only_preempted: bool) -> None:
        prefill_slot_ids: list[int] = []
        prefill_required_pages: list[int] = []

        for id_ in self._empty_slots:

            if self.preempted:
                req = self.preempted.pop()

            elif not only_preempted:
                if wait:
                    req = self.requests.get()
                    wait = False

                else:
                    try:
                        req = self.requests.get_nowait()
                    except queue.Empty:
                        break

            else:
                break

            if req is self.SENTINEL:
                self.open = False
                break

            required_pages = self._required_pages(
                len(req.prompt) + len(req.new_tokens),
                total=False
            )

            if required_pages > len(self._empty_pages) - sum(prefill_required_pages):
                self.preempted.append(req)
                break


            self.slot_listing_order.append(id_)
            self.slots[id_] = req

            prefill_required_pages.append(required_pages)
            prefill_slot_ids.append(id_)

        if prefill_slot_ids:

            self._palloc(
                slot_ids=prefill_slot_ids,
                page_counts=prefill_required_pages
            )

            bucket = self._get_bucket(
                batch_size=len(prefill_slot_ids)
            )

            next_tokens = bucket.prefill_module.forward(
                prompts=[self.slots[id_].prompt + self.slots[id_].new_tokens
                         for id_ in prefill_slot_ids],
                slot_ids=prefill_slot_ids
            )

            self._refresh_slots(
                slot_ids=prefill_slot_ids,
                next_tokens=next_tokens
            )

    def _pre_decode_palloc(self) -> None:
        ps = self._decode_page_requiring_slots
        ep = self._empty_pages

        if len(ps) > len(ep):
            self._free_n_pages(
                len(ps) - len(ep)
            )

        ps = self._decode_page_requiring_slots
        self._palloc(ps, [1] * len(ps))

    def _loop_decode(self) -> None:
        slot_ids = self._filled_slots

        bucket = self._get_bucket(
            batch_size=len(slot_ids)
        )
        next_tokens = bucket.decode_module.forward(
            slot_ids=slot_ids
        )

        self._refresh_slots(
            slot_ids=slot_ids,
            next_tokens=next_tokens
        )

    @property
    def _decode_page_requiring_slots(self) -> list[int]:
        slots = []
        for i in self._filled_slots:
            if self.slots[i].requires_page(self.page_size):
                slots.append(i)

        return slots

    def _free_n_pages(self, n: int) -> None:

        slots = []
        reqs = []
        while n > 0:
            last_slot_id = self.slot_listing_order.pop()
            req = self.slots[last_slot_id]

            n -= len(req.pages)

            slots.append(last_slot_id)
            reqs.append(req)
            self.preempted.append(req)

        self._pfree(slots)

        for req in reqs:
            req.kv_len = 0
            req.pages = set()
            req.prefilled = False

    def _consume_queue(self, exc: Exception) -> None:
        while True:
            try:
                req = self.requests.get_nowait()
            except queue.Empty:
                break

            if req is self.SENTINEL:
                self.open = False
                continue

            req.future.set_exception(exc)

    def _consume_preempted(self, exc: Exception) -> None:
        while self.preempted:
            req = self.preempted.pop()
            req.future.set_exception(exc)

    def _clear(self, ids: list[int] | None = None) -> None:

        if ids is None:
            self.slots = [
                None for _ in range(self.n_slots)
            ]
            self.slot_listing_order.clear()
            self.kv_cache.reset(list(self.slot_ids))

        else:
            for id_ in ids:
                self.slots[id_] = None
                if id_ in self.slot_listing_order:
                    self.slot_listing_order.remove(id_)
            self.kv_cache.reset(ids)

    def _end(self, exc: Exception) -> None:
        for id_ in self._filled_slots:
            self.slots[id_].future.set_exception(exc)

        self._consume_queue(exc=exc)
        self._consume_preempted(exc=exc)
        self._clear()

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

            elif self._required_pages(l, total=True) > self.n_pages - 1:
                fut.set_exception(ValueError(
                    "Possible amount of tokens exceeds available pages."
                ))

            else:
                self.requests.put(Request(
                    prompt=prompt,
                    future=fut,
                    t_arrival=t_arrival
                ))

            futs.append(fut)

        return futs

    def stop(self) -> None:
        self.requests.put(self.SENTINEL)

    def loop(self) -> None:

        while True:
            try:
                if self.open:
                    if self._filled_slots:
                        self._pre_decode_palloc()
                        self._loop_prefill(False, only_preempted=False)

                    else:
                        self._loop_prefill(True, only_preempted=False)

                    if self._filled_slots:
                        self._pre_decode_palloc()
                        self._loop_decode()

                elif self._filled_slots or self.preempted:
                    self._pre_decode_palloc()
                    self._loop_prefill(False, only_preempted=True)
                    self._loop_decode()

                else:
                    exc = RuntimeError(
                        "Loop has been terminated."
                    )
                    self._end(exc=exc)
                    break

            except Exception as exc:
                self._end(exc=exc)