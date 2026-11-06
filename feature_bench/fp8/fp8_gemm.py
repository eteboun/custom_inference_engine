import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from common import MODEL_ID
from feature_bench.common import save_report
from model_loader import build_model
from feature_bench.token_threshold.gemm_token_steps import first_layer_projections, measure_replay, profile_kernels
from quant.core import get_scale, quantize


RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOKEN_COUNTS = [16, 32, 64, 128, 129, 256]
VARIANTS = ["fp16", "fp8_gemm_only", "fp8_fixed_scale", "fp8_dynamic_scale"]
WARMUP_COMPILED_CALLS = 3


class ProjectionModule:

    def __init__(
            self,
            variant: str,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
            token_count: int
    ) -> None:
        self.variant = variant
        self.weight = weight
        self.bias = bias
        self.in_features = weight.shape[1]

        with torch.inference_mode():
            self.input = torch.randn(
                (1, token_count, self.in_features),
                dtype=weight.dtype,
                device=weight.device
            )

            if variant != "fp16":
                self.weight_scale = get_scale(weight)
                self.weight_fp8 = quantize(weight, self.weight_scale)
                self.input_scale = get_scale(self.input)
                self.input_fp8 = quantize(self.input, self.input_scale)

            forward = torch.compile(self._forward, fullgraph=True, dynamic=False)
            for _ in range(WARMUP_COMPILED_CALLS):
                forward()

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = forward()

    def _multiply(self, input_scale: torch.Tensor, input_fp8: torch.Tensor) -> torch.Tensor:
        return torch._scaled_mm(
            input_fp8.view(-1, self.in_features),
            self.weight_fp8.t(),
            scale_a=input_scale,
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=self.input.dtype
        ).unsqueeze(0)

    def _forward(self) -> torch.Tensor:
        if self.variant == "fp16":
            return F.linear(self.input, self.weight, self.bias)
        if self.variant == "fp8_gemm_only":
            return self._multiply(self.input_scale, self.input_fp8)
        if self.variant == "fp8_fixed_scale":
            return self._multiply(self.input_scale, quantize(self.input, self.input_scale))
        input_scale = get_scale(self.input)
        return self._multiply(input_scale, quantize(self.input, input_scale))

    def replay(self) -> None:
        self.graph.replay()

    def relative_error(self) -> float:
        with torch.inference_mode():
            self.replay()
            torch.cuda.synchronize()
            reference = F.linear(
                self.input.float(),
                self.weight.float(),
                None if self.bias is None else self.bias.float()
            )
            return ((self.output.float() - reference).norm() / reference.norm()).item()


def build_modules(token_count: int) -> dict[tuple[str, str], ProjectionModule]:
    torch._dynamo.config.recompile_limit = 256
    torch.manual_seed(0)

    model = build_model(MODEL_ID, torch.float16)
    return {
        (name, variant): ProjectionModule(
            variant=variant,
            weight=weight,
            bias=bias,
            token_count=token_count
        )
        for name, (weight, bias) in first_layer_projections(model).items()
        for variant in VARIANTS
    }


def run_worker(token_count: int, worker_out: Path) -> None:
    modules = build_modules(token_count)

    replay_times = {key: measure_replay(module) for key, module in modules.items()}

    with open(worker_out, "a") as f:
        for (name, variant), module in modules.items():
            kernels = profile_kernels(module)
            replay_us, replay_sd = replay_times[(name, variant)]
            row = {
                "projection": name, "variant": variant, "T": token_count,
                "replay_us": replay_us, "replay_sd": replay_sd,
                "kernel_us": sum(k["us"] for k in kernels.values()),
                "relative_error": module.relative_error(),
                "kernels": kernels,
            }
            f.write(json.dumps(row) + "\n")
            print(
                f"{name} {variant} T={token_count} replay={replay_us:.1f} us "
                f"kernel={row['kernel_us']:.1f} us error={row['relative_error']:.4f}",
                flush=True
            )


def build_tables(rows: list[dict]) -> list[dict]:
    by_key = {(r["projection"], r["variant"], r["T"]): r for r in rows}
    projections = list(dict.fromkeys(r["projection"] for r in rows))
    token_counts = sorted({r["T"] for r in rows})

    def per_token_count(title: str, field: str) -> dict:
        return {
            "title": title,
            "columns": ["projection", "variant", *token_counts],
            "rows": [
                [p, v, *(by_key[(p, v, t)][field] for t in token_counts)]
                for p in projections
                for v in VARIANTS
            ],
        }

    speedup_rows = [
        [
            p, v,
            *(by_key[(p, "fp16", t)]["replay_us"] / by_key[(p, v, t)]["replay_us"] for t in token_counts)
        ]
        for p in projections
        for v in VARIANTS[1:]
    ]

    kernel_rows = []
    for p in projections:
        for v in VARIANTS:
            for t in token_counts:
                kernels = by_key[(p, v, t)]["kernels"]
                for name in sorted(kernels, key=lambda n: -kernels[n]["us"]):
                    kernel = kernels[name]
                    kernel_rows.append([
                        p, v, t, name, kernel["calls"], kernel["us"], kernel.get("grid"), kernel.get("block")
                    ])

    return [
        {
            "title": "Replay speedup over fp16",
            "columns": ["projection", "variant", *token_counts],
            "rows": speedup_rows,
        },
        per_token_count("Replay time per call (us)", "replay_us"),
        per_token_count("Replay time sd (us)", "replay_sd"),
        per_token_count("Total kernel time per call (us)", "kernel_us"),
        per_token_count("Relative error against fp32", "relative_error"),
        {
            "title": "Kernels per call (us)",
            "columns": ["projection", "variant", "T", "kernel", "calls", "us", "grid", "block"],
            "rows": kernel_rows,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default=None)
    parser.add_argument("--token-counts", type=int, nargs="+", default=TOKEN_COUNTS)
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
                    sys.executable, "-m", "feature_bench.fp8.fp8_gemm",
                    "--worker", str(token_count),
                    "--worker-out", str(worker_out),
                ],
                check=True,
            )

        with open(worker_out) as f:
            rows = [json.loads(line) for line in f]

    path = save_report(
        name="fp8_gemm",
        title=args.title,
        settings={
            "token_counts": args.token_counts,
            "variants": VARIANTS,
        },
        tables=build_tables(rows),
        results_dir=args.results_dir
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
