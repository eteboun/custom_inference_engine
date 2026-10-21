import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from generation.common import MODEL_ID, tokenize, get_eos_token_ids
from main import IMPL_FNS

# nsys ile profillemek için küçük config
# manuel decode CUDA graph replay olarak çalışır, graph içindeki kernel'leri
# ayrı ayrı görmek için nsys'e --cuda-graph-trace=node verilmeli.
# graph capture warmup sırasında olur, ölçülen çağrılarda sadece replay'ler var
PROFILE_CONFIG = {
    "max_new_tokens": 8,
    "warmup_iters": 1,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impls", nargs="+", choices=IMPL_FNS.keys(), default=["vllm"]
    )
    impl_names = parser.parse_args().impls

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16
    ).to("cuda").eval()

    messages = [
        {"role": "user", "content": "What is a GPU warp?"}
    ]

    max_new_tokens = PROFILE_CONFIG["max_new_tokens"]
    warmup_iters = PROFILE_CONFIG["warmup_iters"]

    inputs = tokenize(tokenizer, messages, model.device)
    eos_token_ids = get_eos_token_ids(model, tokenizer)

    impls = {name: IMPL_FNS[name] for name in impl_names}

    for _ in range(warmup_iters):
        for generate_fn in impls.values():
            generate_fn(model, inputs, eos_token_ids, max_new_tokens)
    torch.cuda.synchronize()

    for name, generate_fn in impls.items():
        ids = generate_fn(model, inputs, eos_token_ids, max_new_tokens)
        print(f"{name}: {tokenizer.decode(ids, skip_special_tokens=True)!r}")
    torch.cuda.synchronize()

    print(f"seq_len={inputs['input_ids'].shape[-1]}  "
          f"max_new_tokens={max_new_tokens}  warmup_iters={warmup_iters}")
