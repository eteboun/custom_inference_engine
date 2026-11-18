import time

import torch

from kv_cache.storage import KVStorage
from model_loader import build_model
from scheduler.graph_backend import FullGraphBackend, PiecewiseGraphBackend
from scheduler.models import ComponentInitStats, CudaGraphExecutorStats, InitStats


class CudaGraphExecutor:
    def __init__(
        self,
        model_id: str,
        dtype: torch.dtype,
        kv_cache_capacity: int,
        n_slots: int,
        page_size: int,
        batch_sizes: list[int],
        full_token_counts: list[int],
        piecewise_token_counts: list[int],
        quantization: str | None = None,
    ) -> None:
        self.full_token_threshold = max(full_token_counts)
        self.init_stats = InitStats()
        start = time.perf_counter()
        allocated_before = torch.cuda.memory_allocated()
        self.model = build_model(model_id, dtype, quantization)
        self.init_stats.model = self._measure_init(start, allocated_before)
        self.init_stats.calibration_seconds = self.model.calibration_seconds

        start = time.perf_counter()
        allocated_before = torch.cuda.memory_allocated()
        self.kv_cache = KVStorage(
            config=self.model.config,
            max_cache_len=kv_cache_capacity,
            n_slots=n_slots,
            page_size=page_size,
            device=self.model.device,
            dtype=dtype,
        )
        self.init_stats.kv_cache = self._measure_init(start, allocated_before)
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit,
            len(batch_sizes) * len(full_token_counts) + 2 * len(piecewise_token_counts),
        )
        self.graph_pool = torch.cuda.graph_pool_handle()
        start = time.perf_counter()
        allocated_before = torch.cuda.memory_allocated()
        self.piecewise_backend = PiecewiseGraphBackend(
            model=self.model,
            kv_cache=self.kv_cache,
            token_counts=piecewise_token_counts,
            pool=self.graph_pool,
        )
        self.init_stats.piecewise = self._measure_init(start, allocated_before)

        start = time.perf_counter()
        allocated_before = torch.cuda.memory_allocated()
        self.full_backend = FullGraphBackend(
            model=self.model,
            kv_cache=self.kv_cache,
            batch_sizes=batch_sizes,
            token_counts=full_token_counts,
            pool=self.graph_pool,
        )
        self.init_stats.full = self._measure_init(start, allocated_before)

    @staticmethod
    def _measure_init(start: float, allocated_before: int) -> ComponentInitStats:
        allocated = torch.cuda.memory_allocated()
        return ComponentInitStats(
            seconds=time.perf_counter() - start,
            memory=allocated - allocated_before,
            memory_allocated=allocated,
            memory_reserved=torch.cuda.memory_reserved(),
        )

    def execute(
        self,
        slot_ids: list[int],
        prompts: list[list[int]] | None,
        run_stats: CudaGraphExecutorStats,
    ) -> list[int]:
        if not slot_ids:
            return []

        batch_size = len(slot_ids)
        run_stats.batch_sizes[batch_size] += 1
        if prompts is None:
            module = self.full_backend.get_decode_module(batch_size)
            run_stats.decode_steps[module.batch_size] += 1
            return module.forward(slot_ids=slot_ids)

        token_count = sum(map(len, prompts))
        if token_count <= self.full_token_threshold:
            module = self.full_backend.get_mixed_module(batch_size, token_count)
            run_stats.full_mixed_steps[(module.batch_size, module.token_count)] += 1
        else:
            module = self.piecewise_backend.get_mixed_module(token_count)
            run_stats.piecewise_mixed_steps[module.token_count] += 1

        run_stats.tokens += token_count
        run_stats.bucket_tokens += module.token_count
        return module.forward(prompts=prompts, slot_ids=slot_ids)
