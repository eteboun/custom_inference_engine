import argparse
import collections
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from feature_bench.common import TOKEN_THRESHOLD_RESULTS_DIR, load_model, save_report
from feature_bench.html_report import render_report


TOKEN_COUNTS = [128, 129, 130, 131, 132]
WARMUP_REPLAYS = 20
WARMUP_COMPILED_CALLS = 3
TIMING_ITERS = 500
TIMING_REPEATS = 7
PROFILE_ITERS = 100


class GemmModule:

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, token_count: int) -> None:
        self.weight = weight
        self.bias = bias

        with torch.inference_mode():
            self.input = torch.randn(
                (1, token_count, weight.shape[1]),
                dtype=weight.dtype,
                device=weight.device
            )

            forward = torch.compile(self._forward, fullgraph=True, dynamic=False)
            for _ in range(WARMUP_COMPILED_CALLS):
                forward()

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = forward()

    def _forward(self) -> torch.Tensor:
        return F.linear(self.input, self.weight, self.bias)

    def replay(self) -> None:
        self.graph.replay()


def build_modules(token_count: int) -> dict[str, GemmModule]:
    torch._dynamo.config.recompile_limit = 256
    torch.manual_seed(0)

    model = load_model()
    layer = model.model.layers[0]
    attention = layer.self_attn

    projections = {
        "qkv": (attention.qkv_weight, attention.qkv_bias),
        "o_proj": (attention.original.o_proj.weight, attention.original.o_proj.bias),
        "gate_proj": (layer.mlp.gate_proj.weight, layer.mlp.gate_proj.bias),
        "up_proj": (layer.mlp.up_proj.weight, layer.mlp.up_proj.bias),
        "down_proj": (layer.mlp.down_proj.weight, layer.mlp.down_proj.bias),
    }
    return {
        name: GemmModule(weight=weight, bias=bias, token_count=token_count)
        for name, (weight, bias) in projections.items()
    }


def warm_up(module: GemmModule) -> None:
    for _ in range(WARMUP_REPLAYS):
        module.replay()
    torch.cuda.synchronize()


def measure_replay(module: GemmModule) -> tuple[float, float]:
    with torch.inference_mode():
        warm_up(module)

        samples = []
        for _ in range(TIMING_REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(TIMING_ITERS):
                module.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / TIMING_ITERS)

    return statistics.median(samples), statistics.pstdev(samples)


def read_trace_events(prof: profile) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        with open(trace_path) as f:
            return json.load(f)["traceEvents"]


def profile_kernels(module: GemmModule) -> dict[str, dict[str, float]]:
    with torch.inference_mode():
        warm_up(module)

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(PROFILE_ITERS):
                module.replay()
            torch.cuda.synchronize()

    kernels = collections.defaultdict(lambda: {"us": 0.0, "calls": 0.0})
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name]["us"] += event.device_time / PROFILE_ITERS
            kernels[event.name]["calls"] += 1 / PROFILE_ITERS

    for event in read_trace_events(prof):
        if event.get("cat") == "kernel" and event["name"] in kernels:
            kernels[event["name"]]["grid"] = event["args"].get("grid")
            kernels[event["name"]]["block"] = event["args"].get("block")

    return dict(kernels)


def run_worker(token_count: int, worker_out: Path) -> None:
    modules = build_modules(token_count)

    replay_times = {name: measure_replay(module) for name, module in modules.items()}

    with open(worker_out, "a") as f:
        for name, module in modules.items():
            kernels = profile_kernels(module)
            replay_us, replay_sd = replay_times[name]
            row = {
                "projection": name, "T": token_count,
                "shape": list(module.weight.shape),
                "replay_us": replay_us, "replay_sd": replay_sd,
                "kernel_us": sum(k["us"] for k in kernels.values()),
                "kernels": kernels,
            }
            f.write(json.dumps(row) + "\n")
            print(f"{name} T={token_count} replay={replay_us:.1f} us kernel={row['kernel_us']:.1f} us", flush=True)


def build_tables(rows: list[dict]) -> list[dict]:
    by_key = {(r["projection"], r["T"]): r for r in rows}
    projections = list(dict.fromkeys(r["projection"] for r in rows))
    token_counts = sorted({r["T"] for r in rows})

    def per_token_count(title: str, field: str) -> dict:
        return {
            "title": title,
            "columns": ["projection", *token_counts],
            "rows": [
                [p, *(by_key[(p, t)][field] for t in token_counts)]
                for p in projections
            ],
        }

    kernel_rows = []
    for p in projections:
        for t in token_counts:
            kernels = by_key[(p, t)]["kernels"]
            for name in sorted(kernels, key=lambda n: -kernels[n]["us"]):
                kernel = kernels[name]
                kernel_rows.append([
                    p, t, name, kernel["calls"], kernel["us"], kernel.get("grid"), kernel.get("block")
                ])

    return [
        {
            "title": "Weight shapes",
            "columns": ["projection", "out", "in"],
            "rows": [[p, *by_key[(p, token_counts[0])]["shape"]] for p in projections],
        },
        per_token_count("Total kernel time per call (us)", "kernel_us"),
        per_token_count("Replay time per call (us)", "replay_us"),
        per_token_count("Replay time sd (us)", "replay_sd"),
        {
            "title": "Kernels per call (us)",
            "columns": ["projection", "T", "kernel", "calls", "us", "grid", "block"],
            "rows": kernel_rows,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default=None)
    parser.add_argument("--token-counts", type=int, nargs="+", default=TOKEN_COUNTS)
    parser.add_argument("--results-dir", type=Path, default=TOKEN_THRESHOLD_RESULTS_DIR)
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
                    sys.executable, "-m", "feature_bench.gemm_token_steps",
                    "--worker", str(token_count),
                    "--worker-out", str(worker_out),
                ],
                check=True,
            )

        with open(worker_out) as f:
            rows = [json.loads(line) for line in f]

    path = save_report(
        name="gemm_token_steps",
        title=args.title,
        settings={
            "token_counts": args.token_counts,
            "timing_iters": TIMING_ITERS,
            "timing_repeats": TIMING_REPEATS,
            "profile_iters": PROFILE_ITERS,
        },
        tables=build_tables(rows),
        results_dir=args.results_dir
    )
    print(f"saved: {path}")

    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    output = path.with_suffix(".html")
    with open(output, "w", encoding="utf-8") as f:
        f.write(render_report(report))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
