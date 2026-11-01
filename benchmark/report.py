import dataclasses
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from benchmark.metrics import OverallMetrics, RequestMetrics, TimedResult
from benchmark.workload import WorkloadInfo
from feature_bench.html_report import render_report
from scheduler.models import ComponentInitStats, InitStats, RunStats


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MIB = 2 ** 20


@dataclasses.dataclass
class ServeResult:
    output: list[TimedResult]
    init: InitStats
    run: RunStats | None = None


def git_state() -> dict[str, str | bool]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=RESULTS_DIR.parent,
            capture_output=True,
            text=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def environment(engine: str) -> dict[str, str | None]:
    env = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    if engine == "vllm":
        import vllm
        env["vllm"] = vllm.__version__
    return env


def latency_table(summary: OverallMetrics) -> dict:
    def milliseconds(metrics: RequestMetrics, name: str) -> float | None:
        value = getattr(metrics, name)
        return None if value is None else value * 1000

    return {
        "title": "Latency (ms)",
        "columns": ["metric", "mean", "p50", "p99"],
        "rows": [
            [
                field.name,
                milliseconds(summary.means, field.name),
                milliseconds(summary.p50, field.name),
                milliseconds(summary.p99, field.name),
            ]
            for field in dataclasses.fields(RequestMetrics)
        ],
    }


def throughput_table(summary: OverallMetrics) -> dict:
    return {
        "title": "Throughput",
        "columns": ["duration (s)", "requests/s", "tokens/s"],
        "rows": [[summary.duration, summary.req_per_s, summary.tok_per_s]],
    }


def reference_tables(summary: OverallMetrics, n_requests: int) -> list[dict]:
    match = summary.reference_match
    return [
        {
            "title": "Reference match",
            "columns": ["exact requests", "total requests", "mean match ratio"],
            "rows": [[match.n_exact, n_requests, match.mean_match_ratio]],
        },
        {
            "title": "First differing token of non-exact requests",
            "columns": ["token range start", "requests"],
            "rows": [[start, count] for start, count in match.first_diff_histogram.items()],
        },
    ]


def init_table(init: InitStats) -> dict:
    def row(name: str, stats: ComponentInitStats) -> list:
        return [
            name, stats.seconds, stats.memory / MIB,
            stats.memory_allocated / MIB, stats.memory_reserved / MIB,
        ]

    rows = [["total", init.total_seconds, None, None, None]]
    for field in dataclasses.fields(InitStats):
        stats = getattr(init, field.name)
        if isinstance(stats, ComponentInitStats):
            rows.append(row(field.name, stats))

    return {
        "title": "Init",
        "columns": ["component", "seconds", "memory (MiB)", "allocated (MiB)", "reserved (MiB)"],
        "rows": rows,
    }


def run_tables(run: RunStats) -> list[dict]:
    steps = [
        *(["decode", str(bucket), count] for bucket, count in sorted(run.decode_steps.items())),
        *(["full mixed", str(bucket), count] for bucket, count in sorted(run.full_mixed_steps.items())),
        *(["piecewise mixed", str(bucket), count] for bucket, count in sorted(run.piecewise_mixed_steps.items())),
    ]
    return [
        {
            "title": "Steps per backend bucket",
            "columns": ["backend", "bucket", "steps"],
            "rows": steps,
        },
        {
            "title": "Steps per real batch size",
            "columns": ["batch size", "steps"],
            "rows": [[size, count] for size, count in sorted(run.batch_sizes.items())],
        },
        {
            "title": "Scheduler counters",
            "columns": ["mixed tokens", "mixed bucket tokens", "preemptions", "recomputed tokens"],
            "rows": [[run.tokens, run.bucket_tokens, run.preemptions, run.recomputed_tokens]],
        },
    ]


def build_tables(summary: OverallMetrics, serve_result: ServeResult) -> list[dict]:
    tables = [
        throughput_table(summary),
        latency_table(summary),
        *reference_tables(summary, len(serve_result.output)),
        init_table(serve_result.init),
    ]
    if serve_result.run is not None:
        tables += run_tables(serve_result.run)
    return tables


def save_report(
        title: str,
        engine: str,
        settings: dict,
        info: WorkloadInfo,
        summary: OverallMetrics,
        serve_result: ServeResult
) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(exist_ok=True)
    now = datetime.now()
    name = f"{now:%Y%m%d-%H%M%S}_{engine}_{info.kind}_rate{settings['rate']:g}_r{settings['repeat']}"
    json_path = RESULTS_DIR / f"{name}.json"

    report = {
        "title": title,
        "benchmark": "serving",
        "timestamp": now.isoformat(timespec="seconds"),
        "git": git_state(),
        "env": environment(engine),
        "settings": {
            "engine": engine,
            **settings,
            "n_short": info.n_short,
            "n_long": info.n_long,
            "mean_prompt_tokens": info.mean_tokens,
        },
        "tables": build_tables(summary, serve_result),
        "records": [
            {"tokens": tokens, "t_arrival": t_arrival, "t_first": t_first, "t_end": t_end, "planned": planned}
            for (tokens, t_arrival, t_first, t_end), planned in serve_result.output
        ],
    }
    with open(json_path, "w") as f:
        json.dump(report, f)

    html_path = json_path.with_suffix(".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_report(report))

    return json_path, html_path
