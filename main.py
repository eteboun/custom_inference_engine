import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from common import (
    MODEL_ID, tokenize, get_eos_token_ids, run_benchmarks, summarize,
    stats_header, stats_row, eos_cell
)
from vllm_baseline.vllm_generate import generate_vllm


IMPL_LABELS = {
    "vllm": "vLLM (LLMEngine, varsayılan ayarlar)",
}


IMPL_FNS = {
    "vllm": generate_vllm,
}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16
    ).to("cuda").eval()

    messages = [
        {"role": "user", "content": "What is a GPU warp?"}
    ]

    max_new_tokens = 50
    warmup_iters = 3
    iters = 10
    seed = 0

    inputs = tokenize(tokenizer, messages, model.device)
    eos_token_ids = get_eos_token_ids(model, tokenizer)

    results = run_benchmarks(
        model,
        {name: (fn, inputs) for name, fn in IMPL_FNS.items()},
        eos_token_ids, max_new_tokens, warmup_iters, iters, seed
    )

    ref_ids = results["vllm"]["ids"]
    print(f"text: {tokenizer.decode(ref_ids, skip_special_tokens=True)!r}")
    print()
    print(f"{'impl':<28}{'same':>6}{'consistent':>12}  label")
    for name, label in IMPL_LABELS.items():
        r = results[name]
        print(f"{name:<28}{str(r['ids'] == ref_ids):>6}{str(r['consistent']):>12}  {label}")
    print()
    print(f"seq_len={inputs['input_ids'].shape[-1]}  max_new_tokens={max_new_tokens}  "
          f"warmup_iters={warmup_iters}  iters={iters}  seed={seed}  (ms)")
    for metric in ["ttft", "tpot"]:
        print()
        print(f"{'impl':<28}{'new_tok':>9}{'eos':>7}{stats_header(metric)}")
        for name in IMPL_LABELS:
            r = results[name]
            print(
                f"{name:<28}{len(r['ids']):>9}{eos_cell(r):>7}"
                f"{stats_row(summarize(r[metric]))}"
            )
