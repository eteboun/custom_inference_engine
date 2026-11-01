import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from feature_bench.common import PAGE_SIZE, RESULTS_DIR, assign_pages, build_kv, load_model, save_report
from scheduler.graph_backend import PiecewiseGraphBackend
from scheduler.mixed_module import FullMixedModule


WARMUP_REPLAYS = 5


def timed(module, prompts, slot_ids, iters: int, repeats: int) -> tuple[float, float]:
    with torch.inference_mode():
        module.upload(prompts, slot_ids)
        for _ in range(WARMUP_REPLAYS):
            module.replay()
        torch.cuda.synchronize()

        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                module.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / iters)

    return statistics.median(samples), statistics.pstdev(samples)


def run_worker(args) -> None:
    token_count = args.worker
    batch_sizes = [b for b in args.batch_sizes if b <= token_count]
    n_slots = max(args.batch_sizes) + 1

    torch._dynamo.config.recompile_limit = 256

    model = load_model()
    kv = build_kv(model, n_slots)
    pool = torch.cuda.graph_pool_handle()

    piecewise = PiecewiseGraphBackend(model, kv, [token_count], pool)
    full = {
        b: FullMixedModule(model, kv, batch_size=b, token_count=token_count, pool=pool)
        for b in sorted(batch_sizes, reverse=True)
    }

    pages_per_slot = math.ceil((args.context + max(args.token_counts)) / PAGE_SIZE)
    assign_pages(kv, list(range(1, n_slots)), pages_per_slot, args.context)

    with open(args.worker_out, "a") as f:
        for b in batch_sizes:
            lens = [token_count // b + (1 if i < token_count % b else 0) for i in range(b)]
            prompts = [[1000 + i] * n for i, n in enumerate(lens)]
            slot_ids = list(range(1, b + 1))

            full_ms, full_sd = timed(full[b], prompts, slot_ids, args.iters, args.repeats)
            piecewise_ms, piecewise_sd = timed(
                piecewise.get_mixed_module(token_count), prompts, slot_ids, args.iters, args.repeats
            )

            row = {
                "B": b, "T": token_count,
                "full_ms": full_ms, "full_sd": full_sd,
                "piecewise_ms": piecewise_ms, "piecewise_sd": piecewise_sd,
            }
            f.write(json.dumps(row) + "\n")
            print(row, flush=True)


def build_tables(rows: list[dict]) -> list[dict]:
    overhead = {
        (r["T"], r["B"]): 100 * (r["piecewise_ms"] / r["full_ms"] - 1)
        for r in rows
    }
    batch_sizes = sorted({r["B"] for r in rows})
    token_counts = sorted({r["T"] for r in rows})

    return [
        {
            "title": "Piecewise overhead over full graph (%)",
            "columns": ["T \\ B", *batch_sizes],
            "rows": [
                [t, *(overhead.get((t, b)) for b in batch_sizes)]
                for t in token_counts
            ],
        },
        {
            "title": "Replay time (ms)",
            "columns": ["T", "B", "full", "full sd", "piecewise", "piecewise sd", "overhead %"],
            "rows": [
                [
                    r["T"], r["B"], r["full_ms"], r["full_sd"],
                    r["piecewise_ms"], r["piecewise_sd"], overhead[(r["T"], r["B"])],
                ]
                for r in sorted(rows, key=lambda r: (r["T"], r["B"]))
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--token-counts", type=int, nargs="+", required=True)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--worker-out", default=None)
    args = parser.parse_args()

    if args.worker is not None:
        run_worker(args)
        return

    if args.title is None:
        parser.error("--title is required")

    with tempfile.TemporaryDirectory() as tmp:
        worker_out = Path(tmp) / "rows.jsonl"
        for token_count in sorted(args.token_counts, reverse=True):
            subprocess.run(
                [
                    sys.executable, "-m", "feature_bench.step_time",
                    "--batch-sizes", *map(str, args.batch_sizes),
                    "--token-counts", *map(str, args.token_counts),
                    "--context", str(args.context),
                    "--iters", str(args.iters),
                    "--repeats", str(args.repeats),
                    "--worker", str(token_count),
                    "--worker-out", str(worker_out),
                ],
                check=True,
            )

        with open(worker_out) as f:
            rows = [json.loads(line) for line in f]

    path = save_report(
        "step_time",
        args.title,
        {
            "batch_sizes": args.batch_sizes,
            "token_counts": args.token_counts,
            "context": args.context,
            "iters": args.iters,
            "repeats": args.repeats,
        },
        build_tables(rows),
        args.results_dir
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
