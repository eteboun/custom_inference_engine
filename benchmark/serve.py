import argparse
import asyncio
import threading
import time
from typing import Protocol

import torch

from benchmark.client import benchmark
from benchmark.report import ServeResult
from common import MODEL_ID
from scheduler.models import InitStats, RunStats
from scheduler.scheduler import Scheduler


class Server(Protocol):
    def serve(self, requests: list[list[int]], arrivals: list[float]) -> ServeResult:
        ...

    def close(self) -> None:
        ...


class OursServer:

    def __init__(self, args: argparse.Namespace, eos_token_ids: set[int]) -> None:
        self.engine = Scheduler(
            model_id=MODEL_ID,
            eos_token_ids=eos_token_ids,
            max_new_tokens=args.max_new_tokens,
            n_slots=args.n_slots,
            page_size=args.page_size,
            token_budget=args.token_budget,
            dtype=torch.float16
        )
        self.thread = threading.Thread(target=self.engine.loop, daemon=True)
        self.thread.start()

    def serve(self, requests: list[list[int]], arrivals: list[float]) -> ServeResult:
        self.engine.tracker.run = RunStats()

        output = asyncio.run(
            benchmark(
                engine=self.engine,
                requests=requests,
                arrivals=arrivals
            )
        )

        return ServeResult(
            output=output,
            init=self.engine.tracker.init,
            run=self.engine.tracker.run
        )

    def close(self) -> None:
        self.engine.stop()
        self.thread.join()


class VllmServer:

    def __init__(self, args: argparse.Namespace, eos_token_ids: set[int]) -> None:
        from vllm_baseline.vllm_generate import build_vllm_engine

        start = time.perf_counter()
        self.engine = build_vllm_engine(
            max_num_seqs=args.n_slots - 1,
            max_num_batched_tokens=args.token_budget
        )
        self.init = InitStats(total_seconds=time.perf_counter() - start)

        self.eos_token_ids = eos_token_ids
        self.max_new_tokens: int = args.max_new_tokens
        self.n_runs = 0

    def serve(self, requests: list[list[int]], arrivals: list[float]) -> ServeResult:
        from vllm_baseline.vllm_serve import serve_vllm

        output = serve_vllm(
            engine=self.engine,
            requests=requests,
            arrivals=arrivals,
            eos_token_ids=self.eos_token_ids,
            max_new_tokens=self.max_new_tokens,
            tag=str(self.n_runs)
        )
        self.n_runs += 1

        return ServeResult(output=output, init=self.init)

    def close(self) -> None:
        pass
