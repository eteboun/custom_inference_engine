import queue
import time
import math
from concurrent.futures import Future, InvalidStateError

import torch
from typing import ClassVar
from scheduler.executor import CudaGraphExecutor
from scheduler.models import Request, RequestResult, Tracker


class Scheduler:

    DECODE_BATCH_BUCKETS: ClassVar[tuple] = (1, 2, 4, 8, 16, 32)
    KV_CACHE_CAPACITY: ClassVar[int] = 32768
    MIN_TOKEN_BUCKET: ClassVar[int] = 16
    FULL_GRAPH_TOKEN_THRESHOLD: ClassVar[int] = 128
    PIECEWISE_TOKEN_STEP: ClassVar[int] = 32

    SENTINEL: ClassVar[object] = object()

    def __init__(
            self,
            model_id: str,
            eos_token_ids: set[int],
            max_new_tokens: int,
            n_slots: int,
            page_size: int,
            token_budget: int,
            dtype: torch.dtype,
            quantization: str | None = None
    ) -> None:

        init_start = time.perf_counter()

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

        if token_budget <= n_slots - 1:
            raise ValueError(
                "token_budget must be greater than n_slots - 1."
            )

        if token_budget <= self.FULL_GRAPH_TOKEN_THRESHOLD:
            raise ValueError(
                "token_budget must be greater than FULL_GRAPH_TOKEN_THRESHOLD."
            )

        if token_budget % self.PIECEWISE_TOKEN_STEP != 0:
            raise ValueError(
                "token_budget must be a multiple of PIECEWISE_TOKEN_STEP."
            )

        self.max_new_tokens = max_new_tokens
        self.n_slots = n_slots
        self.page_size = page_size
        self.token_budget = token_budget
        self.eos_token_ids = frozenset(eos_token_ids)

        self.n_pages = math.ceil(self.KV_CACHE_CAPACITY / page_size)
        if self._required_pages(1, True) > self.n_pages - 1:
            raise ValueError(
                f"max_new_tokens can't need more than {self.n_pages - 1} pages."
            )

        self.executor = CudaGraphExecutor(
            model_id=model_id,
            dtype=dtype,
            kv_cache_capacity=self.KV_CACHE_CAPACITY,
            n_slots=n_slots,
            page_size=page_size,
            batch_sizes=self._batch_buckets(),
            full_token_counts=self._full_token_buckets(),
            piecewise_token_counts=self._piecewise_token_buckets(),
            quantization=quantization,
        )
        self.kv_cache = self.executor.kv_cache
        self.tracker = Tracker(init=self.executor.init_stats)

        self.requests = queue.Queue()
        self.preempted: list[Request] = []

        self.slot_listing_order: list[int] = []
        self.slots: list[Request | None] = [
            None for _ in range(self.n_slots)
        ]

        self.page_ids = set(range(self.n_pages))
        self.slot_ids = set(range(self.n_slots))
        self._reset_occupancy()
        self.open = True

        self.tracker.init.total_seconds = time.perf_counter() - init_start

    def _reset_occupancy(self) -> None:
        self._empty_pages: set[int] = self.page_ids - {self.kv_cache.unused_idx}
        self._empty_slots: set[int] = self.slot_ids - {self.kv_cache.unused_idx}
        self._filled_slots: set[int] = set()

    def _batch_buckets(self) -> list[int]:
        sizes = []
        for size in self.DECODE_BATCH_BUCKETS:
            sizes.append(size)
            if size >= self.n_slots - 1:
                break

        return sizes

    def _full_token_buckets(self) -> list[int]:
        sizes = []
        size = self.MIN_TOKEN_BUCKET
        while size < self.FULL_GRAPH_TOKEN_THRESHOLD:
            sizes.append(size)
            size *= 2

        sizes.append(self.FULL_GRAPH_TOKEN_THRESHOLD)
        return sizes

    def _piecewise_token_buckets(self) -> list[int]:
        return list(range(
            self.FULL_GRAPH_TOKEN_THRESHOLD + self.PIECEWISE_TOKEN_STEP,
            self.token_budget + self.PIECEWISE_TOKEN_STEP,
            self.PIECEWISE_TOKEN_STEP
        ))

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
    ) -> None:

        if len(page_counts) != len(slot_ids):
            raise RuntimeError(
                "Unmatching page counts."
            )

        if any(id_ < 0 or id_ == self.kv_cache.unused_idx or id_ >= self.n_slots for id_ in slot_ids):
            raise RuntimeError(
                "Invalid slot ids."
            )

        if any(
                self.slots[id_].prefilled
                and page_counts[i] > 0
                and not self.slots[id_].requires_page(self.page_size)
                for i, id_ in enumerate(slot_ids)
        ):
            raise RuntimeError(
                "Cannot allocate pages to slots with unfilled last page."
            )

        if any(
                count < 0
                or count > self.n_pages - 1 - len(self.slots[id_].pages)
                for id_, count in zip(slot_ids, page_counts)
        ):
            raise RuntimeError(
                "Invalid page counts."
            )

        if sum(page_counts) > len(self._empty_pages):
            raise RuntimeError(
                "No enough pages to allocate."
            )

    def _palloc(
            self,
            slot_ids: list[int],
            page_counts: list[int]
    ) -> None:

        self._validate_palloc(
            slot_ids,
            page_counts
        )

        for id_, page_count in zip(slot_ids, page_counts):
            if page_count == 0:
                continue

            req = self.slots[id_]

            next_ = len(req.pages)
            piece = [self._empty_pages.pop() for _ in range(page_count)]

            self.kv_cache.block_table[
                id_, next_:next_ + page_count
            ] = torch.tensor(piece, device=self.kv_cache.device)

            req.pages.update(piece)

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

            req.kv_len += req.budget

            if not req.prefilled and req.kv_len == len(req.prompt) + len(req.new_tokens):
                req.prefilled = True

            if not req.prefilled:
                continue

            if (
                    token in self.eos_token_ids
                    or len(req.new_tokens) == self.max_new_tokens - 1
            ):
                self._complete(
                    req.future,
                    RequestResult(
                        new_tokens=req.new_tokens + [token],
                        t_arrival=req.t_arrival,
                        t_first=req.t_first or current,
                        t_end=current
                    )
                )

                finished_slots.append(slot_id)

            else:
                if not req.new_tokens:
                    req.t_first = current
                    req.new_tokens = [token]

                else:
                    req.new_tokens.append(token)


        self._pfree(finished_slots)

    def _preempt_slots(self, slot_ids: list[int]) -> None:
        reqs = [self.slots[id_] for id_ in slot_ids]
        self._clear(slot_ids)

        self.tracker.run.scheduler.preemptions += len(reqs)

        for req in reqs:
            self.tracker.run.scheduler.recomputed_tokens += req.kv_len
            req.prefilled = False
            req.kv_len = 0
            req.pages = set()
            req.budget = 0

            self.preempted.append(req)

    def _preempt_accepted_request(self, req: Request) -> None:
        self.preempted.append(req)

    def _accept_requests(
            self,
            wait: bool,
            only_preempted: bool,
            rem_budget: int
    ) -> tuple[
        list[int], list[int], list[int]
    ]:
        new_slot_ids: list[int] = []
        new_required_pages: list[int] = []
        new_budgets: list[int] = []

        for id_ in sorted(self._empty_slots):

            if rem_budget <= 0:
                break

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

            prompt_len = len(req.prompt) + len(req.new_tokens)
            required_budget = min(rem_budget, prompt_len)

            required_pages = self._required_pages(
                required_budget,
                total=False
            )

            if required_pages > len(self._empty_pages) - sum(new_required_pages):
                self._preempt_accepted_request(req)
                break

            self.slot_listing_order.append(id_)
            self.slots[id_] = req
            self._empty_slots.discard(id_)
            self._filled_slots.add(id_)

            new_slot_ids.append(id_)
            new_required_pages.append(required_pages)
            new_budgets.append(required_budget)

            rem_budget -= required_budget

        return new_slot_ids, new_required_pages, new_budgets

    @staticmethod
    def _complete(future: Future, result: RequestResult) -> None:
        try:
            future.set_result(result)
        except InvalidStateError:
            pass

    @staticmethod
    def _fail(future: Future, exc: Exception) -> None:
        try:
            future.set_exception(exc)
        except InvalidStateError:
            pass

    def _consume_queue(self, exc: Exception) -> None:
        while True:
            try:
                req = self.requests.get_nowait()
            except queue.Empty:
                break

            if req is self.SENTINEL:
                self.open = False
                continue

            self._fail(req.future, exc)

    def _consume_preempted(self, exc: Exception) -> None:
        while self.preempted:
            req = self.preempted.pop()
            self._fail(req.future, exc)

    def _clear(self, ids: list[int] | None = None) -> None:

        if ids is None:
            self.slots = [
                None for _ in range(self.n_slots)
            ]
            self.slot_listing_order.clear()
            self.kv_cache.reset(list(self.slot_ids))
            self._reset_occupancy()

        else:
            for id_ in ids:
                req = self.slots[id_]
                if req is not None:
                    self._empty_pages.update(req.pages)

                self.slots[id_] = None
                self._filled_slots.discard(id_)
                self._empty_slots.add(id_)
                if id_ in self.slot_listing_order:
                    self.slot_listing_order.remove(id_)
            self.kv_cache.reset(ids)

    def _end(self, exc: Exception) -> None:
        for id_ in self._filled_slots:
            self._fail(self.slots[id_].future, exc)

        self._consume_queue(exc=exc)
        self._consume_preempted(exc=exc)
        self._clear()

    def call(self, prompt: list[int]) -> Future:
        t_arrival = time.perf_counter()

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

        return fut

    def stop(self) -> None:
        self.requests.put(self.SENTINEL)

    def loop(self) -> None:

        while True:
            try:
                if self.open:
                    if self._filled_slots:
                        rem_budget = self._allocate_pages_pre_request()
                        new = self._accept_requests(
                            wait=False,
                            only_preempted=False,
                            rem_budget=rem_budget
                        )
                        self._allocate_pages_post_request(*new)

                    else:
                        new = self._accept_requests(
                            wait=True,
                            only_preempted=False,
                            rem_budget=self.token_budget
                        )
                        self._allocate_pages_post_request(*new)

                    self._process_slots()

                elif self._filled_slots or self.preempted:
                    rem_budget = self._allocate_pages_pre_request()
                    new = self._accept_requests(
                        wait=False,
                        only_preempted=True,
                        rem_budget=rem_budget
                    )
                    self._allocate_pages_post_request(*new)

                    self._process_slots()

                else:
                    exc = RuntimeError(
                        "Loop has been terminated."
                    )
                    self._end(exc=exc)
                    break

            except Exception as exc:
                self._end(exc=exc)

    def _allocate_pages_pre_request(self) -> int:

        slot_ids = []
        page_counts = []
        budgets = []

        current_budget = self.token_budget
        for i in self.slot_listing_order:

            slot_ids.append(i)

            if current_budget <= 0:
                budgets.append(0)
                page_counts.append(0)
                continue

            req = self.slots[i]

            if req.prefilled:

                if req.requires_page(self.page_size):
                    page_counts.append(1)

                else:
                    page_counts.append(0)

                current_budget -= 1
                budgets.append(1)

            else:
                allocated_budget = min(current_budget, req.uncached_token_count())
                required_page_count = math.ceil(
                    (req.kv_len + allocated_budget) / self.page_size
                ) - len(req.pages)

                page_counts.append(required_page_count)

                current_budget -= allocated_budget
                budgets.append(allocated_budget)

        freed_page_count = 0
        freed_slots = []
        while len(self._empty_pages) + freed_page_count < sum(page_counts):

            slot_id = slot_ids.pop()

            freed_page_count += len(self.slots[slot_id].pages)
            freed_slots.append(slot_id)

            page_counts.pop()
            current_budget += budgets.pop()

        self._preempt_slots(
            slot_ids=freed_slots
        )
        self._palloc(
            slot_ids=slot_ids,
            page_counts=page_counts
        )

        for i, id_ in enumerate(slot_ids):
            self.slots[id_].budget = budgets[i]

        return current_budget

    def _allocate_pages_post_request(
            self,
            slot_ids: list[int],
            page_counts: list[int],
            budgets: list[int]
    ) -> None:
        for i, id_ in enumerate(slot_ids):
            self.slots[id_].budget = budgets[i]

        self._palloc(
            slot_ids=slot_ids,
            page_counts=page_counts
        )

    def _process_slots(self) -> None:
        input_slots = [
            slot_id for slot_id in self._filled_slots if self.slots[slot_id].budget > 0
        ]
        if not input_slots:
            return

        prompts = None
        if any(not self.slots[slot_id].prefilled for slot_id in input_slots):
            prompts = []
            for slot_id in input_slots:
                req = self.slots[slot_id]
                full_prompt = req.prompt + req.new_tokens
                prompts.append(full_prompt[req.kv_len: req.kv_len + req.budget])

        next_tokens = self.executor.execute(input_slots, prompts, self.tracker.run.executor)
        self._refresh_slots(slot_ids=input_slots, next_tokens=next_tokens)
