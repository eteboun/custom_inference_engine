import argparse
import random
from pathlib import Path

import torch
import torch._dynamo.utils as dynamo_utils

from feature_bench.common import RESULTS_DIR, assign_pages, build_kv, load_model, save_report
from scheduler.graph_backend import PiecewiseGraphBackend
from scheduler.mixed_module import FullMixedModule


N_SLOTS = 9
PAGES_PER_SLOT = 64
TOKEN_COUNTS = [160, 192, 224, 256]

CASES = {
    "B=8, 256": [[40, 30, 50, 20, 36, 30, 25, 25]],
    "B=2, 150": [[100, 50]],
    "B=5, 200": [[60, 50, 40, 30, 20]],
    "B=1, 250": [[250]],
    "B=1, 129": [[129]],
    "B=3, 2 chunk": [[90, 60, 40], [70, 80, 50]],
    "B=1, 3 chunk": [[200], [180], [256]],
    "B=4, 2 chunk": [[50, 50, 50, 50], [1, 1, 1, 200]],
}


def unique_graphs() -> int:
    return dynamo_utils.counters["stats"]["unique_graphs"]


def step(module, prompts, slot_ids) -> torch.Tensor:
    with torch.inference_mode():
        module.upload(prompts, slot_ids)
        hidden = module.replay()
        return module._result(hidden)[:len(slot_ids)].float().clone()


def run_steps(kv, get_module, steps, slot_ids) -> list[torch.Tensor]:
    assign_pages(kv, slot_ids, PAGES_PER_SLOT)
    return [
        step(get_module(sum(map(len, prompts))), prompts, slot_ids)
        for prompts in steps
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    torch._dynamo.config.recompile_limit = 64

    model = load_model()
    kv = build_kv(model, N_SLOTS)
    pool = torch.cuda.graph_pool_handle()

    piecewise = PiecewiseGraphBackend(model, kv, TOKEN_COUNTS, pool)
    full = {
        t: FullMixedModule(model, kv, batch_size=N_SLOTS - 1, token_count=t, pool=pool)
        for t in reversed(TOKEN_COUNTS)
    }

    def get_full(t):
        return full[next(x for x in TOKEN_COUNTS if x >= t)]

    rng = random.Random(args.seed)
    rows = []
    all_same = True
    for name, chunks in CASES.items():
        steps = [
            [[rng.randrange(100, 30000) for _ in range(n)] for n in lens]
            for lens in chunks
        ]
        slot_ids = list(range(1, len(chunks[0]) + 1))

        before = unique_graphs()
        expected = run_steps(kv, get_full, steps, slot_ids)
        actual = run_steps(kv, piecewise.get_mixed_module, steps, slot_ids)
        new_graphs = unique_graphs() - before

        for i, (x, y) in enumerate(zip(expected, actual)):
            same = torch.equal(x.argmax(-1), y.argmax(-1))
            all_same &= same
            rows.append([
                name, i, sum(chunks[i]), (x - y).abs().max().item(), same, new_graphs
            ])
            print(rows[-1], flush=True)

    kv.reset(list(range(N_SLOTS)))

    path = save_report(
        "piecewise_vs_full",
        args.title,
        {
            "n_slots": N_SLOTS,
            "token_counts": TOKEN_COUNTS,
            "seed": args.seed,
            "all_argmax_same": all_same,
        },
        [{
            "title": "Piecewise vs full graph logits",
            "columns": ["case", "step", "T", "max |dlogit|", "argmax same", "new graphs"],
            "rows": rows,
        }],
        args.results_dir
    )
    print("ALL ARGMAX SAME" if all_same else "MISMATCH")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
