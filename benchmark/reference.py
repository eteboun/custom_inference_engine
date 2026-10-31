import hashlib
import json

import torch
from transformers import AutoModelForCausalLM

from common import MODEL_ID
from benchmark.report import RESULTS_DIR


REFERENCE_DIR = RESULTS_DIR / "reference"


def load_reference(prompts, eos_token_ids, max_new_tokens):
    key = hashlib.sha256(
        json.dumps([prompts, sorted(eos_token_ids), max_new_tokens]).encode()
    ).hexdigest()[:16]
    path = REFERENCE_DIR / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to("cuda").eval()

    eos = sorted(eos_token_ids)
    refs = []
    with torch.inference_mode():
        for prompt in prompts:
            input_ids = torch.tensor([prompt], device="cuda")
            out = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                top_k=None,
                top_p=None,
                temperature=None,
                eos_token_id=eos,
                pad_token_id=eos[0],
            )
            refs.append(out[0, len(prompt):].tolist())

    del model
    torch.cuda.empty_cache()

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(refs, f)
    return refs
