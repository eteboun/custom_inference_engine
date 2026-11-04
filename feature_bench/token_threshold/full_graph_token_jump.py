import argparse
import collections
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from feature_bench.common import assign_pages, build_kv, load_model, save_report
from feature_bench.token_threshold.step_time import WARMUP_REPLAYS, timed
from kv_cache.storage import KVStorage
from scheduler.mixed_module import FullMixedModule


RESULTS_DIR = Path(__file__).resolve().parent / "results"
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
TOKEN_COUNTS = [128, 132]
CONTEXT = 256
PAGES_PER_SLOT = 26
TIMING_ITERS = 50
TIMING_REPEATS = 7
PROFILE_ITERS = 20


def build_modules(token_count: int) -> tuple[KVStorage, dict[int, FullMixedModule]]:
    torch._dynamo.config.recompile_limit = 256

    model = load_model()
    kv = build_kv(model, n_slots=max(BATCH_SIZES) + 1)
    pool = torch.cuda.graph_pool_handle()

    modules = {
        batch_size: FullMixedModule(
            model=model,
            kv_cache=kv,
            batch_size=batch_size,
            token_count=token_count,
            pool=pool
        )
        for batch_size in sorted(BATCH_SIZES, reverse=True)
    }
    return kv, modules


def make_inputs(batch_size: int, token_count: int) -> tuple[list[list[int]], list[int]]:
    lens = [
        token_count // batch_size + (1 if i < token_count % batch_size else 0)
        for i in range(batch_size)
    ]
    prompts = [[1000 + i] * n for i, n in enumerate(lens)]
    slot_ids = list(range(1, batch_size + 1))
    return prompts, slot_ids


def profile_kernels(
        module: FullMixedModule,
        prompts: list[list[int]],
        slot_ids: list[int]
) -> dict[str, dict[str, float]]:
    with torch.inference_mode():
        module.upload(prompts, slot_ids)
        for _ in range(WARMUP_REPLAYS):
            module.replay()
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(PROFILE_ITERS):
                module.replay()
            torch.cuda.synchronize()

    kernels = collections.defaultdict(lambda: {"us": 0.0, "calls": 0.0})
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name]["us"] += event.device_time / PROFILE_ITERS
            kernels[event.name]["calls"] += 1 / PROFILE_ITERS
    return dict(kernels)


def run_worker(token_count: int, worker_out: Path) -> None:
    kv, modules = build_modules(token_count)

    assign_pages(
        kv,
        slot_ids=list(range(1, kv.n_slots)),
        pages_per_slot=PAGES_PER_SLOT,
        context=CONTEXT
    )

    inputs = {b: make_inputs(b, token_count) for b in BATCH_SIZES}

    replay_times = {
        b: timed(
            modules[b],
            prompts=inputs[b][0],
            slot_ids=inputs[b][1],
            iters=TIMING_ITERS,
            repeats=TIMING_REPEATS
        )
        for b in BATCH_SIZES
    }

    with open(worker_out, "a") as f:
        for b in BATCH_SIZES:
            kernels = profile_kernels(modules[b], prompts=inputs[b][0], slot_ids=inputs[b][1])
            replay_ms, replay_sd = replay_times[b]
            row = {
                "B": b, "T": token_count,
                "replay_ms": replay_ms, "replay_sd": replay_sd,
                "kernel_ms": sum(k["us"] for k in kernels.values()) / 1000,
                "kernels": kernels,
            }
            f.write(json.dumps(row) + "\n")
            print(f"B={b} T={token_count} replay={replay_ms:.3f} ms kernel={row['kernel_ms']:.3f} ms", flush=True)


def build_tables(rows: list[dict]) -> list[dict]:
    low, high = sorted({r["T"] for r in rows})
    by_key = {(r["B"], r["T"]): r for r in rows}

    summary_rows = []
    for b in BATCH_SIZES:
        a, c = by_key[(b, low)], by_key[(b, high)]
        delta_replay = c["replay_ms"] - a["replay_ms"]
        delta_kernel = c["kernel_ms"] - a["kernel_ms"]
        summary_rows.append([
            b, a["replay_ms"], a["replay_sd"], c["replay_ms"], c["replay_sd"], delta_replay,
            a["kernel_ms"], c["kernel_ms"], delta_kernel, delta_replay - delta_kernel,
        ])

    tables = [{
        "title": "Replay and total kernel time (ms)",
        "columns": [
            "B", f"replay {low}", f"replay {low} sd", f"replay {high}", f"replay {high} sd", "delta replay",
            f"kernel {low}", f"kernel {high}", "delta kernel", "delta replay - delta kernel",
        ],
        "rows": summary_rows,
    }]

    for b in BATCH_SIZES:
        low_kernels = by_key[(b, low)]["kernels"]
        high_kernels = by_key[(b, high)]["kernels"]

        kernel_rows = []
        for name in low_kernels.keys() | high_kernels.keys():
            a, c = low_kernels.get(name), high_kernels.get(name)
            a_us = a["us"] if a else 0.0
            c_us = c["us"] if c else 0.0
            kernel_rows.append([
                name,
                a["calls"] if a else None, a["us"] if a else None,
                c["calls"] if c else None, c["us"] if c else None,
                c_us - a_us,
            ])
        kernel_rows.sort(key=lambda row: abs(row[-1]), reverse=True)

        tables.append({
            "title": f"Kernels per replay, B={b} (us)",
            "columns": ["kernel", f"calls {low}", f"us {low}", f"calls {high}", f"us {high}", "delta us"],
            "rows": kernel_rows,
        })

    return tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default=None)
    parser.add_argument("--token-counts", type=int, nargs=2, default=TOKEN_COUNTS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--worker-out", type=Path, default=None)
    args = parser.parse_args()

    if args.worker is not None:
        run_worker(args.worker, worker_out=args.worker_out)
        return

    if args.title is None:
        parser.error("--title is required")

    with tempfile.TemporaryDirectory() as tmp:
        worker_out = Path(tmp) / "rows.jsonl"
        for token_count in sorted(args.token_counts, reverse=True):
            subprocess.run(
                [
                    sys.executable, "-m", "feature_bench.token_threshold.full_graph_token_jump",
                    "--worker", str(token_count),
                    "--worker-out", str(worker_out),
                ],
                check=True,
            )

        with open(worker_out) as f:
            rows = [json.loads(line) for line in f]

    path = save_report(
        name="full_graph_token_jump",
        title=args.title,
        settings={
            "batch_sizes": BATCH_SIZES,
            "token_counts": args.token_counts,
            "context": CONTEXT,
            "pages_per_slot": PAGES_PER_SLOT,
            "timing_iters": TIMING_ITERS,
            "timing_repeats": TIMING_REPEATS,
            "profile_iters": PROFILE_ITERS,
        },
        tables=build_tables(rows),
        results_dir=args.results_dir
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
