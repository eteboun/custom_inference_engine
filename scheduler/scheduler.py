import queue
import time
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

        max_cache_len = self.KV_CACHE_CAPACITY // n_slots

        if max_new_tokens > max_cache_len:
            raise ValueError(
                f"max_new_tokens can't be larger than max_cache_len {max_cache_len}."
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
        self.max_cache_len = max_cache_len
        self.n_slots = n_slots
        self.eos_token_ids = frozenset(eos_token_ids)

        self.kv_cache = KVStorage(
            config=model.config,
            max_cache_len=max_cache_len,
            n_slots=n_slots,
            device=model.device,
            dtype=dtype
        )
        self.buckets = self._generate_buckets()
        self.requests = queue.Queue()

        self.futs: dict[int, Future | None] = {
            i: None
            for i in range(self.n_slots)
        }

        self.new_tokens: dict[int, list[int]] = {}
        self.t_arrivals: dict[int, float] = {}
        self.t_firsts: dict[int, float] = {}

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

    def _empty_slots(self) -> list[int]:
        return [
            i
            for i in self.futs
            if i != 0 and self.futs[i] is None
        ]

    def _filled_slots(self) -> list[int]:
        return [
            i
            for i in self.futs
            if i != 0 and self.futs[i] is not None
        ]

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
                    or len(self.new_tokens.get(slot_id, [])) == self.max_new_tokens - 1
            ):
                t_arrival = self.t_arrivals.pop(slot_id)
                t_first = self.t_firsts.pop(slot_id, current)
                new_tokens = self.new_tokens.pop(
                    slot_id, []
                ) + [token]

                self.futs[slot_id].set_result(
                    (new_tokens, t_arrival, t_first, current)
                )

                self.futs[slot_id] = None
                finished_slots.append(slot_id)

            else:
                if slot_id not in self.new_tokens:
                    self.t_firsts[slot_id] = current
                    self.new_tokens[slot_id] = [token]

                else:
                    self.new_tokens[slot_id].append(token)

        self.kv_cache.reset(finished_slots)

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

    def _clear(self) -> None:
        self.futs = {
            i: None
            for i in range(self.n_slots)
        }
        self.new_tokens = {}
        self.t_firsts = {}
        self.t_arrivals = {}

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

            elif l + self.max_new_tokens > self.max_cache_len:
                fut.set_exception(ValueError(
                    "Possible amount of tokens exceeds max_cache_len."
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
