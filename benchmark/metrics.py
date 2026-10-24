import math
import statistics


METRIC_COLUMNS = [
    ("ttft", "mean"), ("ttft", "p50"), ("ttft", "p99"),
    ("tpot", "mean"), ("tpot", "p50"), ("tpot", "p99"),
    ("e2e", "mean"), ("e2e", "p99"),
]


def percentile(samples, q):
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    k = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def describe(samples):
    return {
        "mean": statistics.mean(samples) if samples else float("nan"),
        "p50": percentile(samples, 50),
        "p99": percentile(samples, 99),
    }


def compute_metrics(records):
    start = min(r["planned"] for r in records)
    end = max(r["t_end"] for r in records)
    duration = end - start
    total_tokens = sum(len(r["tokens"]) for r in records)

    ttft = [r["t_first"] - r["planned"] for r in records]
    e2e = [r["t_end"] - r["planned"] for r in records]
    tpot = [
        (r["t_end"] - r["t_first"]) / (len(r["tokens"]) - 1)
        for r in records if len(r["tokens"]) > 1
    ]

    return {
        "n_requests": len(records),
        "duration": duration,
        "req_per_s": len(records) / duration,
        "tok_per_s": total_tokens / duration,
        "mean_new_tokens": total_tokens / len(records),
        "ttft": describe(ttft),
        "tpot": describe(tpot),
        "e2e": describe(e2e),
    }


def table_header():
    cols = "".join(f"{m + '_' + s:>11}" for m, s in METRIC_COLUMNS)
    return f"{'rate':>6}{'req/s':>8}{'tok/s':>9}{'new_tok':>9}{cols}"


def table_row(rate, m):
    cols = "".join(f"{m[metric][stat] * 1000:>11.2f}" for metric, stat in METRIC_COLUMNS)
    return (
        f"{rate:>6g}{m['req_per_s']:>8.2f}{m['tok_per_s']:>9.1f}"
        f"{m['mean_new_tokens']:>9.1f}{cols}"
    )
