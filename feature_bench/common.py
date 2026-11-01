import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from attn.core import use_triton_attention
from benchmark.report import environment, git_state
from common import MODEL_ID
from kv_cache.storage import KVStorage


RESULTS_DIR = Path(__file__).resolve().parent / "results"
PAGE_SIZE = 16
KV_CAPACITY = 32768


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to("cuda").eval()
    use_triton_attention(model)
    return model


def build_kv(model, n_slots: int) -> KVStorage:
    return KVStorage(
        config=model.config,
        max_cache_len=KV_CAPACITY,
        n_slots=n_slots,
        page_size=PAGE_SIZE,
        device=model.device,
        dtype=torch.float16
    )


def assign_pages(kv: KVStorage, slot_ids: list[int], pages_per_slot: int, context: int = 0) -> None:
    kv.reset(list(range(kv.n_slots)))
    for slot in slot_ids:
        start = 1 + (slot - 1) * pages_per_slot
        kv.block_table[slot, :pages_per_slot] = torch.arange(
            start, start + pages_per_slot, device=kv.device
        )
        kv.lens[slot] = context


def save_report(name: str, title: str, settings: dict, tables: list[dict], results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    path = results_dir / f"{now:%Y%m%d-%H%M%S}_{name}.json"
    with open(path, "w") as f:
        json.dump({
            "title": title,
            "benchmark": name,
            "timestamp": now.isoformat(timespec="seconds"),
            "git": git_state(),
            "env": environment("ours"),
            "settings": settings,
            "tables": tables,
        }, f)
    return path
