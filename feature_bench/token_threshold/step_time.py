import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from common import MODEL_ID
from feature_bench.common import PAGE_SIZE, assign_pages, build_kv, save_report
from model_loader import build_model
from scheduler.graph_backend import PiecewiseGraphBackend
from scheduler.mixed_module import FullMixedModule, MixedModule


RESULTS_DIR = Path(__file__).resolve().parent / "results"
WARMUP_REPLAYS = 5


def timed(
        module: MixedModule,
        prompts: list[list[int]],
        slot_ids: list[int],
        iters: int,
        repeats: int
) -> tuple[float, float]:
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


def run_worker(args: argparse.Namespace) -> None:
    token_count = args.worker
    full_bucket = args.full_bucket or token_count
    piecewise_bucket = (
        math.ceil(token_count / args.piecewise_bucket_step) * args.piecewise_bucket_step
        if args.piecewise_bucket_step is not None else token_count
    )
    batch_sizes = [b for b in args.batch_sizes if b <= token_count]
    n_slots = max(args.batch_sizes) + 1

    torch._dynamo.config.recompile_limit = 256

    model = build_model(args.model, torch.float16)
    kv = build_kv(model, n_slots=n_slots)
    pool = torch.cuda.graph_pool_handle()

    piecewise = PiecewiseGraphBackend(model=model, kv_cache=kv, token_counts=[piecewise_bucket], pool=pool)
    full = {
        b: FullMixedModule(model=model, kv_cache=kv, batch_size=b, token_count=full_bucket, pool=pool)
        for b in sorted(batch_sizes, reverse=True)
    }

    pages_per_slot = math.ceil((args.context + max(args.token_counts)) / PAGE_SIZE)
    assign_pages(kv, slot_ids=list(range(1, n_slots)), pages_per_slot=pages_per_slot, context=args.context)

    with open(args.worker_out, "a") as f:
        for b in batch_sizes:
            lens = [token_count // b + (1 if i < token_count % b else 0) for i in range(b)]
            prompts = [[1000 + i] * n for i, n in enumerate(lens)]
            slot_ids = list(range(1, b + 1))

            full_ms, full_sd = timed(
                full[b], prompts=prompts, slot_ids=slot_ids, iters=args.iters, repeats=args.repeats
            )
            piecewise_ms, piecewise_sd = timed(
                piecewise.get_mixed_module(piecewise_bucket),
                prompts=prompts,
                slot_ids=slot_ids,
                iters=args.iters,
                repeats=args.repeats
            )

            row = {
                "B": b, "T": token_count,
                "full_bucket": full_bucket, "piecewise_bucket": piecewise_bucket,
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
            "title": "Piecewise time relative to full graph (%)",
            "columns": ["T \\ B", *batch_sizes],
            "rows": [
                [t, *(overhead.get((t, b)) for b in batch_sizes)]
                for t in token_counts
            ],
        },
        {
            "title": "Replay time (ms)",
            "columns": ["real tokens", "batch", "full bucket", "piecewise bucket",
                        "full", "full sd", "piecewise", "piecewise sd", "difference %"],
            "rows": [
                [
                    r["T"], r["B"], r["full_bucket"], r["piecewise_bucket"],
                    r["full_ms"], r["full_sd"],
                    r["piecewise_ms"], r["piecewise_sd"], overhead[(r["T"], r["B"])],
                ]
                for r in sorted(rows, key=lambda r: (r["T"], r["B"]))
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default=None)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--token-counts", type=int, nargs="+", required=True)
    parser.add_argument("--full-bucket", type=int, help="Use this full graph bucket for every token count.")
    parser.add_argument("--piecewise-bucket-step", type=int, help="Round piecewise graph buckets up to this multiple.")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--worker-out", default=None)
    args = parser.parse_args()
    if args.full_bucket is not None and args.full_bucket < max(args.token_counts):
        parser.error("--full-bucket must cover all real token counts")
    if args.piecewise_bucket_step is not None and args.piecewise_bucket_step <= 0:
        parser.error("--piecewise-bucket-step must be positive")

    if args.worker is not None:
        run_worker(args)
        return

    if args.title is None:
        parser.error("--title is required")

    bucket_args = []
    if args.full_bucket is not None:
        bucket_args += ["--full-bucket", str(args.full_bucket)]
    if args.piecewise_bucket_step is not None:
        bucket_args += ["--piecewise-bucket-step", str(args.piecewise_bucket_step)]

    with tempfile.TemporaryDirectory() as tmp:
        worker_out = Path(tmp) / "rows.jsonl"
        for token_count in sorted(args.token_counts, reverse=True):
            subprocess.run(
                [
                    sys.executable, "-m", "feature_bench.token_threshold.step_time",
                    "--model", args.model,
                    "--batch-sizes", *map(str, args.batch_sizes),
                    "--token-counts", *map(str, args.token_counts),
                    "--context", str(args.context),
                    "--iters", str(args.iters),
                    "--repeats", str(args.repeats),
                    "--worker", str(token_count),
                    "--worker-out", str(worker_out),
                    *bucket_args,
                ],
                check=True,
            )

        with open(worker_out) as f:
            rows = [json.loads(line) for line in f]

    path = save_report(
        name="step_time",
        title=args.title,
        settings={
            "model": args.model,
            "batch_sizes": args.batch_sizes,
            "token_counts": args.token_counts,
            "full_bucket": args.full_bucket,
            "piecewise_bucket_step": args.piecewise_bucket_step,
            "context": args.context,
            "iters": args.iters,
            "repeats": args.repeats,
        },
        tables=build_tables(rows),
        results_dir=args.results_dir
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
