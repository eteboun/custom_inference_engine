import random
import statistics
import time

import torch


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def get_eos_token_ids(model, tokenizer):
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    return set(eos)


def tokenize(tokenizer, messages, device):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)


def measure(generate_fn, model, inputs, eos_token_ids, max_new_tokens):
    token_times = []

    torch.cuda.synchronize()
    start = time.perf_counter()

    ids = generate_fn(
        model, inputs, eos_token_ids, max_new_tokens,
        on_token=lambda: token_times.append(time.perf_counter())
    )

    hit_eos = len(token_times) > len(ids)

    ttft = token_times[0] - start if len(token_times) >= 1 else None
    tpot = (
        (token_times[-1] - token_times[0]) / (len(token_times) - 1)
        if len(token_times) >= 2 else None
    )

    return ids, hit_eos, ttft, tpot


def run_benchmarks(model, cases, eos_token_ids, max_new_tokens,
                   warmup_iters=3, iters=10, seed=0):
    rng = random.Random(seed)

    warmup_schedule = [key for key in cases for _ in range(warmup_iters)]
    rng.shuffle(warmup_schedule)
    for key in warmup_schedule:
        generate_fn, inputs = cases[key]
        generate_fn(model, inputs, eos_token_ids, max_new_tokens)

    schedule = [key for key in cases for _ in range(iters)]
    rng.shuffle(schedule)

    results = {
        key: {"ids": None, "consistent": True, "eos": 0, "runs": 0,
              "ttft": [], "tpot": []}
        for key in cases
    }

    for key in schedule:
        generate_fn, inputs = cases[key]
        ids, hit_eos, ttft, tpot = measure(
            generate_fn, model, inputs, eos_token_ids, max_new_tokens
        )

        if results[key]["ids"] is not None and results[key]["ids"] != ids:
            results[key]["consistent"] = False
        results[key]["ids"] = ids
        results[key]["eos"] += hit_eos
        results[key]["runs"] += 1
        if ttft is not None:
            results[key]["ttft"].append(ttft)
        if tpot is not None:
            results[key]["tpot"].append(tpot)

    return results


def summarize(samples):
    if not samples:
        nan = float("nan")
        return {"mean": nan, "median": nan, "std": nan, "min": nan,
                "p25": nan, "p75": nan, "max": nan}

    if len(samples) >= 2:
        p25, _, p75 = statistics.quantiles(samples, n=4, method="inclusive")
        std = statistics.stdev(samples)
    else:
        p25 = p75 = samples[0]
        std = 0.0

    return {
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "std": std,
        "min": min(samples),
        "p25": p25,
        "p75": p75,
        "max": max(samples),
    }


STAT_KEYS = ["mean", "median", "std", "min", "p25", "p75", "max"]


def eos_cell(result):
    return f"{result['eos']}/{result['runs']}"


def stats_header(metric):
    return "".join(f"{metric + '_' + k:>13}" for k in STAT_KEYS)


def stats_row(stats):
    return "".join(f"{stats[k] * 1000:>13.2f}" for k in STAT_KEYS)
