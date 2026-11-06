import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attn.core import merge_qkv, use_triton_attention
from quant.calibration import calibrate
from quant.core import ALL_PROJECTIONS, use_fp8

QUANTIZATIONS = (None, "fp8")


def build_model(
        model_id: str,
        dtype: torch.dtype,
        quantization: str | None = None,
        fp8_projections: frozenset[str] = ALL_PROJECTIONS
):
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"quantization must be one of {QUANTIZATIONS}.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype
    ).to("cuda").eval()
    model.calibration_seconds = None
    merge_qkv(model)

    if quantization == "fp8":
        use_fp8(model, fp8_projections)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        start = time.perf_counter()
        calibrate(model, tokenizer)
        torch.cuda.synchronize()
        model.calibration_seconds = time.perf_counter() - start

    use_triton_attention(model)
    return model
