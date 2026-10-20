import argparse

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from generation.common import (
    MODEL_ID, tokenize, get_eos_token_ids, run_benchmarks, summarize,
    stats_header, stats_row, eos_cell
)
from main import IMPL_LABELS, IMPL_FNS

FILLER = (
    "A GPU executes threads in groups called warps. "
    "Each warp runs the same instruction on different data. "
)
QUESTION = "\n\nSummarize the text above in detail."


def build_messages(tokenizer, target_seq_len):
    def prompt_len(content):
        return len(tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True
        )["input_ids"])

    overhead = prompt_len(QUESTION)
    n_filler = max(target_seq_len - overhead, 0)

    filler_ids = tokenizer(FILLER)["input_ids"]
    repeats = n_filler // len(filler_ids) + 1
    filler_ids = (filler_ids * repeats)[:n_filler]

    content = tokenizer.decode(filler_ids) + QUESTION
    messages = [{"role": "user", "content": content}]

    return messages


BENCHMARK_CONFIGS = {
    # geliştirme sırasında hızlı kontrol için
    "dev": {
        "target_seq_lens": [64, 1024],
        "max_new_tokens": 16,
        "warmup_iters": 1,
        "iters": 3,
        "seed": 0,
        "plot_path": "results/test_dev.png",
    },
    # raporlanacak sonuçlar için
    "final_benchmark": {
        "target_seq_lens": [64, 256, 1024, 2048, 4096, 8192, 16384],
        "max_new_tokens": 64,
        "warmup_iters": 2,
        "iters": 10,
        "seed": 0,
        "plot_path": "results/test.png",
    },
}


# renkler sabit sırada, legend'e ek olarak her serinin kendi marker'ı var
# (açık renklerin zemine kontrastı düşük, renk tek ayırt edici olmasın)
SERIES_STYLES = {
    "vllm": ("#2a78d6", "o"),
    "manual_inductor_triton_attn": ("#d63a3a", "P"),
}
REFERENCE_IMPL = "vllm"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def plot_results(stats, seq_lens, max_new_tokens, iters, path="results/test.png"):
    # stats: {(name, seq_len): {"ttft": summary, "tpot": summary}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    for ax, metric, title in [
        (axes[0], "ttft", "TTFT (ms)"),
        (axes[1], "tpot", "TPOT (ms)"),
    ]:
        for name, (color, marker) in SERIES_STYLES.items():
            summaries = [stats[(name, n)][metric] for n in seq_lens]
            median = [s["median"] * 1000 for s in summaries]
            p25 = [s["p25"] * 1000 for s in summaries]
            p75 = [s["p75"] * 1000 for s in summaries]

            ax.fill_between(seq_lens, p25, p75, color=color, alpha=0.12, linewidth=0)
            ax.plot(
                seq_lens, median,
                color=color, linewidth=2, marker=marker, markersize=8,
                markeredgecolor="white", markeredgewidth=2,
                solid_capstyle="round", solid_joinstyle="round",
                label=IMPL_LABELS[name]
            )

        ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12)
        # uzunluklar ~2'nin kuvvetleri, doğrusal eksende kısalar üst üste biniyor
        ax.set_xscale("log", base=2)
        ax.set_xticks(seq_lens, [str(n) for n in seq_lens])
        ax.minorticks_off()
        ax.set_xlabel("prompt seq_len (token, log ölçek)", color=TEXT_SECONDARY)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    axes[0].legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    fig.suptitle(
        f"vLLM vs manuel decode  |  max_new_tokens={max_new_tokens}",
        x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13
    )
    fig.text(
        0.01, 0.01,
        f"çizgi: median, bant: p25-p75 ({iters} ölçüm, random sıra)",
        ha="left", color=TEXT_SECONDARY, fontsize=9
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    print()
    print(f"plot: {path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", choices=BENCHMARK_CONFIGS.keys(), default="dev"
    )
    config_name = parser.parse_args().config
    config = BENCHMARK_CONFIGS[config_name]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16
    ).to("cuda").eval()

    target_seq_lens = config["target_seq_lens"]
    max_new_tokens = config["max_new_tokens"]
    warmup_iters = config["warmup_iters"]
    iters = config["iters"]
    seed = config["seed"]

    eos_token_ids = get_eos_token_ids(model, tokenizer)

    cases = {}
    seq_lens = []
    for target in target_seq_lens:
        messages = build_messages(tokenizer, target)
        inputs = tokenize(tokenizer, messages, model.device)
        seq_len = inputs["input_ids"].shape[-1]
        seq_lens.append(seq_len)

        for name, fn in IMPL_FNS.items():
            cases[(name, seq_len)] = (fn, inputs)

    results = run_benchmarks(
        model, cases, eos_token_ids, max_new_tokens, warmup_iters, iters, seed
    )

    stats = {
        key: {metric: summarize(r[metric]) for metric in ["ttft", "tpot"]}
        for key, r in results.items()
    }

    print(f"config={config_name}  max_new_tokens={max_new_tokens}  warmup_iters={warmup_iters}  "
          f"iters={iters}  seed={seed}  (ms)")
    print()
    for name, label in IMPL_LABELS.items():
        print(f"{name:<28}{label}")

    # hız karşılaştırması ancak çıktılar doğruysa anlamlı.
    # same: referansla (REFERENCE_IMPL) aynı token'lar, değilse ilk
    # farklı token'ın indeksi. c: çalıştırmalar arası tutarlı
    print()
    print(f"çıktı karşılaştırması (referans: {REFERENCE_IMPL})")
    print(f"{'seq_len':>8}" + "".join(f"{name:>28}" for name in IMPL_LABELS))
    for seq_len in seq_lens:
        ref = results[(REFERENCE_IMPL, seq_len)]["ids"]
        row = f"{seq_len:>8}"
        for name in IMPL_LABELS:
            r = results[(name, seq_len)]
            if r["ids"] == ref:
                same = "same"
            else:
                first_diff = next(
                    (i for i, (a, b) in enumerate(zip(ref, r["ids"])) if a != b),
                    min(len(ref), len(r["ids"]))
                )
                same = f"diff@{first_diff}"
            row += f"{same + '/' + ('c' if r['consistent'] else 'INCONSISTENT'):>28}"
        print(row)

    for metric in ["ttft", "tpot"]:
        print()
        print(f"{'impl':<28}{'seq_len':>8}{'new_tok':>9}{'eos':>7}{stats_header(metric)}")
        for seq_len in seq_lens:
            for name in IMPL_LABELS:
                key = (name, seq_len)
                print(
                    f"{name:<28}{seq_len:>8}{len(results[key]['ids']):>9}"
                    f"{eos_cell(results[key]):>7}{stats_row(stats[key][metric])}"
                )

    plot_results(stats, seq_lens, max_new_tokens, iters, config["plot_path"])
