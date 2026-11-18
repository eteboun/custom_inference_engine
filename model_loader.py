import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attn.core import use_triton_attention, wrap_qwen2_attention
from quant.calibration import calibrate
from quant.core import ALL_PROJECTIONS, use_fp8

QUANTIZATIONS = (None, "fp8")


def build_model(
        model_id: str,
        dtype: torch.dtype,
        quantization: str | None = None,
        fp8_projections: frozenset[str] = ALL_PROJECTIONS,
        *,
        attention_backend: str = "triton",
):
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"quantization must be one of {QUANTIZATIONS}.")

    if attention_backend not in ("eager", "triton"):
        raise ValueError("attention_backend must be 'eager' or 'triton'.")

    load_kwargs = {}
    if attention_backend == "eager":
        load_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, **load_kwargs
    ).to("cuda").eval()
    model.calibration_seconds = None
    if quantization == "fp8":
        if attention_backend == "triton":
            wrap_qwen2_attention(model, merge_qkv=True)
        use_fp8(model, fp8_projections)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        start = time.perf_counter()
        calibrate(model, tokenizer)
        torch.cuda.synchronize()
        model.calibration_seconds = time.perf_counter() - start

    if attention_backend == "triton":
        use_triton_attention(model)
    return model
