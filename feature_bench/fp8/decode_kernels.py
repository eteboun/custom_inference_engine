import argparse
import collections
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from common import MODEL_ID
from feature_bench.common import assign_pages, build_kv, save_report
from model_loader import build_model
from feature_bench.token_threshold.gemm_token_steps import read_trace_events
from scheduler.decode_module import DecodeModule


RESULTS_DIR = Path(__file__).resolve().parent / "results"
VARIANTS = ["fp16", "fp8"]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
CONTEXT = 256
PAGES_PER_SLOT = 26
WARMUP_REPLAYS = 5
TIMING_ITERS = 50
TIMING_REPEATS = 7
PROFILE_ITERS = 20
PROJECTION_MIN_CALLS = 24


def reset_context(kv, slot_ids: list[int]) -> None:
    kv.lens[slot_ids] = CONTEXT
    torch.cuda.synchronize()


def measure_replay(module: DecodeModule, kv, slot_ids: list[int]) -> tuple[float, float]:
    samples = []
    with torch.inference_mode():
        for _ in range(TIMING_REPEATS):
            reset_context(kv, slot_ids)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(TIMING_ITERS):
                module.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / TIMING_ITERS)

    return statistics.median(samples), statistics.pstdev(samples)


def profile_kernels(module: DecodeModule, kv, slot_ids: list[int]) -> list[dict]:
    with torch.inference_mode():
        reset_context(kv, slot_ids)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(PROFILE_ITERS):
                module.replay()
            torch.cuda.synchronize()

    events = collections.defaultdict(list)
    for event in read_trace_events(prof):
        if event.get("cat") in ("kernel", "gpu_memset"):
            args = event["args"]
            key = (event["name"], str(args.get("grid")), str(args.get("block")))
            events[key].append((event["ts"], event["dur"]))

    kernels = []
    for (name, grid, block), timed_events in events.items():
        durations = [dur for _, dur in sorted(timed_events)]
        calls = len(durations) // PROFILE_ITERS
        per_replay = [
            sum(durations[i * calls:(i + 1) * calls])
            for i in range(PROFILE_ITERS)
        ]
        kernels.append({
            "name": name, "grid": grid, "block": block,
            "calls": calls, "us": statistics.median(per_replay),
        })

    return kernels


def kernel_group(name: str, calls: float) -> str:
    if re.search("gemm|gemv|splitKreduce", name):
        return "matmul, 24+ calls" if calls >= PROJECTION_MIN_CALLS else "matmul, under 24 calls"
    if name.startswith("Memset"):
        return "other"
    if "decode_" in name:
        return "attention"
    return "other"


def build_tables(rows: list[dict]) -> list[dict]:
    groups = ["matmul, 24+ calls", "matmul, under 24 calls", "attention", "other"]

    summary_rows = []
    kernel_rows = []
    for row in rows:
        kernels = row["kernels"]
        totals = collections.Counter()
        for kernel in kernels:
            totals[kernel_group(kernel["name"], kernel["calls"])] += kernel["us"]

        kernel_ms = sum(totals.values()) / 1000
        summary_rows.append([
            row["variant"], row["B"], row["replay_ms"], row["replay_sd"], kernel_ms,
            *(totals[g] / 1000 for g in groups),
        ])

        for kernel in sorted(kernels, key=lambda k: -k["us"]):
            kernel_rows.append([
                row["variant"], row["B"], kernel["name"], kernel_group(kernel["name"], kernel["calls"]),
                kernel["calls"], kernel["us"], kernel["grid"], kernel["block"],
            ])

    calibration = {row["variant"]: row["calibration_s"] for row in rows}

    return [
        {
            "title": "Calibration time (s)",
            "columns": ["variant", "seconds"],
            "rows": [[variant, seconds] for variant, seconds in calibration.items()],
        },
        {
            "title": "Decode replay and kernel time per replay (ms)",
            "columns": ["variant", "B", "replay", "replay sd", "total kernel", *groups],
            "rows": summary_rows,
        },
        {
            "title": "Kernels per replay (us)",
            "columns": ["variant", "B", "kernel", "group", "calls", "us", "grid", "block"],
            "rows": kernel_rows,
        },
    ]


def run_worker(variant: str, worker_out: Path) -> None:
    torch._dynamo.config.recompile_limit = 256
    torch.manual_seed(0)

    model = build_model(MODEL_ID, torch.float16, quantization="fp8" if variant == "fp8" else None)
    kv = build_kv(model, n_slots=max(BATCH_SIZES) + 1)
    pool = torch.cuda.graph_pool_handle()

    modules = {
        b: DecodeModule(model=model, kv_cache=kv, batch_size=b, pool=pool)
        for b in sorted(BATCH_SIZES, reverse=True)
    }

    assign_pages(
        kv,
        slot_ids=list(range(1, kv.n_slots)),
        pages_per_slot=PAGES_PER_SLOT,
        context=CONTEXT
    )

    slot_ids = {b: list(range(1, b + 1)) for b in BATCH_SIZES}
    with torch.inference_mode():
        kv.last_generated[list(range(1, kv.n_slots))] = torch.arange(
            1000, 1000 + kv.n_slots - 1, device=kv.device
        )
        for b in BATCH_SIZES:
            reset_context(kv, slot_ids[b])
            modules[b].upload(slot_ids[b])
            for _ in range(WARMUP_REPLAYS):
                modules[b].replay()
            torch.cuda.synchronize()

    replay_times = {b: measure_replay(modules[b], kv, slot_ids[b]) for b in BATCH_SIZES}

    with open(worker_out, "a") as f:
        for b in BATCH_SIZES:
            replay_ms, replay_sd = replay_times[b]
            row = {
                "variant": variant,
                "calibration_s": model.calibration_seconds,
                "B": b,
                "replay_ms": replay_ms,
                "replay_sd": replay_sd,
                "kernels": profile_kernels(modules[b], kv, slot_ids[b]),
            }
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--worker", choices=VARIANTS, default=None)
    parser.add_argument("--worker-out", type=Path, default=None)
    args = parser.parse_args()

    if args.worker is not None:
        run_worker(args.worker, worker_out=args.worker_out)
        return

    if args.title is None:
        parser.error("--title is required")

    with tempfile.TemporaryDirectory() as tmp:
        worker_out = Path(tmp) / "rows.jsonl"
        for variant in VARIANTS:
            subprocess.run(
                [
                    sys.executable, "-m", "feature_bench.fp8.decode_kernels",
                    "--worker", variant,
                    "--worker-out", str(worker_out),
                ],
                check=True,
            )

        with open(worker_out) as f:
            rows = [json.loads(line) for line in f]

    path = save_report(
        name="decode_kernels",
        title=args.title,
        settings={
            "variants": VARIANTS,
            "batch_sizes": BATCH_SIZES,
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
