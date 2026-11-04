import argparse
import collections
import gc
import math
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from attn.core import merge_projections
from benchmark.reference import load_references
from benchmark.workload import build_workload
from common import MODEL_ID
from feature_bench.common import PAGE_SIZE, assign_pages, build_kv, load_model, save_report
from quant.core import ALL_PROJECTIONS, Fp8Linear, FP8_MAX
from scheduler.mixed_module import MixedModule


RESULTS_DIR = Path(__file__).resolve().parent / "results"
KINDS = ["short", "long"]
VARIANTS = {
    "triton_fp16": frozenset(),
    "fp8_qkv": frozenset({"qkv"}),
    "fp8_o_proj": frozenset({"o_proj"}),
    "fp8_mlp": frozenset({"gate_proj", "up_proj", "down_proj"}),
    "fp8": ALL_PROJECTIONS,
}
SLOT = 1
FP8_MIN_NORMAL = 2 ** -6
PROJECTIONS = ["qkv", "o_proj", "gate_proj", "up_proj", "down_proj"]


class EagerMixedModule(MixedModule):

    def __init__(self, model, kv_cache, token_count: int) -> None:
        super().__init__(model, kv_cache, token_count)
        self.metadata.prepare_full_graph(1)

    def _pad(self, slot_ids: list[int]) -> list[int]:
        return slot_ids

    def _write_batch(self, slot_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
        self.metadata.write_cu_seqlens(cu_seqlens)
        self.metadata.write_slot_ids(slot_ids)

    def replay(self) -> torch.Tensor:
        return self._forward()


def activation_hook(
        module: Fp8Linear,
        args: tuple[torch.Tensor],
        output: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        name: str,
        layer_idx: int,
        results: collections.defaultdict,
):
    x = args[0]
    reference = F.linear(x, weight, bias).float()
    output = output.float()

    scaled_x = x / module.max_inp_s
    results[layer_idx][name].append({
        "max": x.max().item(),
        "min": x.min().item(),
        "pos_clamped": (scaled_x > FP8_MAX).sum().item(),
        "neg_clamped": (scaled_x < -FP8_MAX).sum().item(),
        "coarse": (scaled_x.abs() < FP8_MIN_NORMAL).sum().item(),
        "elements": x.numel(),
        "relative_error": ((output - reference).norm() / reference.norm()).item(),
    })


def set_weight_stats(
        fp8weight: torch.Tensor,
        fp8scale: torch.Tensor,
        fp16weight: torch.Tensor,
        weight_stats: collections.defaultdict,
        name: str,
        layer_idx: int
) -> None:
    weight = fp8weight.float()
    reference = fp16weight.float()
    restored = fp8scale * weight

    weight_stats[layer_idx][name] = {
        "relative_error": ((restored - reference).norm() / reference.norm()).item(),
        "coarse": (weight.abs() < FP8_MIN_NORMAL).sum().item(),
        "zeros": (weight == 0).sum().item(),
        "elements": weight.numel(),
    }


def attach_hooks(
        fp8model,
        fp16model,
        results: collections.defaultdict,
        weight_stats: collections.defaultdict
) -> None:
    with torch.no_grad():
        for layer_idx, fp8layer, fp16layer in zip(
                range(len(fp8model.model.layers)),
                fp8model.model.layers,
                fp16model.model.layers
        ):
            fp8attn = fp8layer.self_attn
            fp16attn = fp16layer.self_attn

            if isinstance(fp8attn.original.o_proj, Fp8Linear):
                fp8attn.original.o_proj.register_forward_hook(
                    partial(activation_hook, weight=fp16attn.o_proj.weight, bias=fp16attn.o_proj.bias,
                            name="o_proj", results=results, layer_idx=layer_idx)
                )

                set_weight_stats(
                    fp8weight=fp8attn.original.o_proj.w,
                    fp8scale=fp8attn.original.o_proj.sw,
                    fp16weight=fp16attn.o_proj.weight,
                    weight_stats=weight_stats,
                    name="o_proj",
                    layer_idx=layer_idx
                )

            if isinstance(fp8attn.qkv, Fp8Linear):
                fp16qkv = merge_projections(
                    [fp16attn.q_proj, fp16attn.k_proj, fp16attn.v_proj]
                )
                fp8attn.qkv.register_forward_hook(
                    partial(activation_hook, weight=fp16qkv.weight, bias=fp16qkv.bias,
                            name="qkv", results=results, layer_idx=layer_idx)
                )

                set_weight_stats(
                    fp8weight=fp8attn.qkv.w,
                    fp8scale=fp8attn.qkv.sw,
                    fp16weight=fp16qkv.weight,
                    weight_stats=weight_stats,
                    name="qkv",
                    layer_idx=layer_idx
                )

            fp8mlp = fp8layer.mlp
            fp16mlp = fp16layer.mlp
            for name in ("gate_proj", "up_proj", "down_proj"):
                fp8linear = getattr(fp8mlp, name)
                fp16linear = getattr(fp16mlp, name)

                if isinstance(fp8linear, Fp8Linear):
                    fp8linear.register_forward_hook(
                        partial(activation_hook, weight=fp16linear.weight, bias=fp16linear.bias,
                                name=name, results=results, layer_idx=layer_idx)
                    )

                    set_weight_stats(
                        fp8weight=fp8linear.w,
                        fp8scale=fp8linear.sw,
                        fp16weight=fp16linear.weight,
                        weight_stats=weight_stats,
                        name=name,
                        layer_idx=layer_idx
                    )


def reference_logits(model, sequence: list[int], start: int) -> torch.Tensor:
    input_ids = torch.tensor([sequence], device=model.device)
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    return model.lm_head(hidden[0, start:]).float()


def engine_logits(model, kv, sequence: list[int], start: int) -> torch.Tensor:
    assign_pages(
        kv,
        slot_ids=[SLOT],
        pages_per_slot=math.ceil(len(sequence) / PAGE_SIZE),
        context=0
    )
    module = EagerMixedModule(model, kv, token_count=len(sequence))
    module.upload(prompts=[sequence], slot_ids=[SLOT])
    hidden = module.replay()
    return model.lm_head(hidden[0, start:]).float()


def compare(reference: torch.Tensor, logits: torch.Tensor) -> dict[str, torch.Tensor]:
    log_p = F.log_softmax(reference, dim=-1)
    log_q = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    q = log_q.exp()

    return {
        "top1": reference.argmax(dim=-1) == logits.argmax(dim=-1),
        "kl": (p * (log_p - log_q)).sum(dim=-1),
        "tv": 0.5 * (p - q).abs().sum(dim=-1),
    }


def first_difference(top1: torch.Tensor) -> int:
    differences = (~top1).nonzero()
    return len(top1) if len(differences) == 0 else differences[0].item()


def summary_row(variant: str, kind: str, results: list[dict[str, torch.Tensor]]) -> list:
    top1 = torch.cat([r["top1"] for r in results]).float()
    kl = torch.cat([r["kl"] for r in results])
    tv = torch.cat([r["tv"] for r in results])
    first = [first_difference(r["top1"]) for r in results]
    exact = sum(bool(r["top1"].all()) for r in results)

    return [
        variant, kind, len(results), len(top1),
        top1.mean().item(),
        kl.mean().item(), kl.quantile(0.99).item(), kl.max().item(),
        tv.mean().item(), tv.quantile(0.99).item(), tv.max().item(),
        sum(first) / len(first), exact,
    ]


def projection_row(name: str, measurements: list[dict]) -> list:
    if not measurements:
        return [name, *[None] * (len(PROJECTION_COLUMNS) - 1)]

    elements = sum(m["elements"] for m in measurements)
    errors = torch.tensor([m["relative_error"] for m in measurements])

    return [
        name, len(measurements), elements,
        max(m["max"] for m in measurements),
        min(m["min"] for m in measurements),
        100 * sum(m["pos_clamped"] for m in measurements) / elements,
        100 * sum(m["neg_clamped"] for m in measurements) / elements,
        100 * sum(m["coarse"] for m in measurements) / elements,
        errors.mean().item(), errors.quantile(0.99).item(), errors.max().item(),
    ]


def weight_row(name: str, stats: list[dict]) -> list:
    if not stats:
        return [name, *[None] * (len(WEIGHT_COLUMNS) - 1)]

    elements = sum(s["elements"] for s in stats)
    errors = [s["relative_error"] for s in stats]

    return [
        name, elements,
        100 * sum(s["coarse"] for s in stats) / elements,
        100 * sum(s["zeros"] for s in stats) / elements,
        sum(errors) / len(errors), max(errors),
    ]


WEIGHT_COLUMNS = [
    "projection", "elements", "below fp8 normal range %", "zero %",
    "relative error mean", "relative error max",
]

PROJECTION_COLUMNS = [
    "projection", "calls", "input elements", "input max", "input min",
    "clamped high %", "clamped low %", "below fp8 normal range %",
    "relative error mean", "relative error p99", "relative error max",
]


def load_eos_token_ids() -> set[int]:
    eos = GenerationConfig.from_pretrained(MODEL_ID).eos_token_id
    return set(eos) if isinstance(eos, list) else {eos}


def run_variant(
        projections: frozenset[str],
        reference_model,
        workloads: dict[str, list]
) -> dict:
    model = load_model(fp8=bool(projections), fp8_projections=projections)

    activations = collections.defaultdict(lambda: collections.defaultdict(list))
    weights = collections.defaultdict(dict)
    attach_hooks(model, reference_model, results=activations, weight_stats=weights)

    kv = build_kv(model, n_slots=SLOT + 1)

    comparisons = {kind: [] for kind in workloads}
    with torch.inference_mode():
        for kind, requests in workloads.items():
            for prompt, new_tokens in requests:
                sequence = prompt + new_tokens[:-1]
                start = len(prompt) - 1
                reference = reference_logits(reference_model, sequence, start)
                logits = engine_logits(model, kv, sequence, start)
                comparisons[kind].append(compare(reference, logits))

    result = {
        "comparisons": comparisons,
        "activations": activations,
        "weights": weights,
        "calibration_seconds": model.calibration_seconds,
    }

    del model, kv
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--n-prompts", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eos_token_ids = load_eos_token_ids()

    workloads = {}
    for kind in KINDS:
        prompts, _ = build_workload(tokenizer, kind=kind, n=args.n_prompts, seed=args.seed)
        tokens = load_references(
            requests=prompts,
            eos_token_ids=eos_token_ids,
            max_new_tokens=args.max_new_tokens
        )
        workloads[kind] = list(zip(prompts, tokens))

    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to("cuda").eval()
    n_layers = len(reference_model.model.layers)

    results = {
        variant: run_variant(projections, reference_model, workloads)
        for variant, projections in VARIANTS.items()
    }

    rows = [
        summary_row(variant, kind, results[variant]["comparisons"][kind])
        for variant in VARIANTS
        for kind in KINDS
    ]

    fp8_variants = [variant for variant, projections in VARIANTS.items() if projections]

    path = save_report(
        name="correctness",
        title=args.title,
        settings={
            "reference": "plain HF fp16",
            "variants": {variant: sorted(projections) for variant, projections in VARIANTS.items()},
            "kinds": KINDS,
            "n_prompts": args.n_prompts,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "calibration_seconds": {
                variant: result["calibration_seconds"] for variant, result in results.items()
            },
        },
        tables=[
            {
                "title": "Teacher-forced comparison against plain HF fp16",
                "columns": [
                    "variant", "kind", "prompts", "positions", "top-1 agreement",
                    "KL mean", "KL p99", "KL max", "TV mean", "TV p99", "TV max",
                    "mean first difference", "prompts with all positions agreeing",
                ],
                "rows": rows,
            },
            {
                "title": "fp8 projections: inputs and error of each projection for the same input",
                "columns": ["variant", *PROJECTION_COLUMNS],
                "rows": [
                    [variant, *projection_row(
                        name,
                        [m for layer in results[variant]["activations"].values() for m in layer[name]]
                    )]
                    for variant in fp8_variants
                    for name in PROJECTIONS
                ],
            },
            {
                "title": "fp8 projections per layer",
                "columns": ["variant", "layer", *PROJECTION_COLUMNS],
                "rows": [
                    [variant, layer, *projection_row(name, results[variant]["activations"][layer][name])]
                    for variant in fp8_variants
                    for layer in range(n_layers)
                    for name in PROJECTIONS
                ],
            },
            {
                "title": "fp8 weights",
                "columns": ["variant", *WEIGHT_COLUMNS],
                "rows": [
                    [variant, *weight_row(
                        name,
                        [layer[name] for layer in results[variant]["weights"].values() if name in layer]
                    )]
                    for variant in fp8_variants
                    for name in PROJECTIONS
                ],
            },
            {
                "title": "fp8 weights per layer",
                "columns": ["variant", "layer", *WEIGHT_COLUMNS],
                "rows": [
                    [variant, layer, *weight_row(
                        name,
                        [results[variant]["weights"][layer][name]] if name in results[variant]["weights"][layer] else []
                    )]
                    for variant in fp8_variants
                    for layer in range(n_layers)
                    for name in PROJECTIONS
                ],
            },
        ],
        results_dir=args.results_dir
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
