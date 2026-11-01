import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from common import MODEL_ID


REFERENCE_DIR = Path(__file__).resolve().parent.parent / "results" / "reference"


def generate_references(
        requests: list[list[int]],
        eos_token_ids: list[int],
        max_new_tokens: int
) -> list[list[int]]:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to("cuda").eval()

    references = []
    with torch.inference_mode():
        for request in requests:
            input_ids = torch.tensor([request], device="cuda")
            output = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                top_k=None,
                top_p=None,
                temperature=None,
                eos_token_id=eos_token_ids,
                pad_token_id=eos_token_ids[0],
            )
            references.append(output[0, len(request):].tolist())

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return references


def load_references(
        requests: list[list[int]],
        eos_token_ids: set[int],
        max_new_tokens: int
) -> list[list[int]]:
    eos = sorted(eos_token_ids)
    key = hashlib.sha256(
        json.dumps([requests, eos, max_new_tokens]).encode()
    ).hexdigest()[:16]
    path = REFERENCE_DIR / f"{key}.json"

    if path.exists():
        with open(path) as f:
            return json.load(f)

    references = generate_references(requests, eos, max_new_tokens)

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(references, f)

    return references
