import dataclasses
import html
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from benchmark.metrics import OverallMetrics, RequestMetrics, TimedResult
from benchmark.workload import WorkloadInfo
from feature_bench.html_report import render_page, render_table
from scheduler.models import ComponentInitStats, InitStats, RunStats


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MIB = 2 ** 20

RunKey = tuple[str, float, int]

RUN_SPECIFIC_SETTINGS = ["kind", "rate", "repeat", "arrival_seed", "trim", "n_short", "n_long", "mean_prompt_tokens"]
SUMMARIZED_TABLES = {"Throughput", "Latency (ms)", "Reference match", "Init"}


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
        "columns": ["measured requests", "duration (s)", "requests/s", "tokens/s"],
        "rows": [[summary.n_measured, summary.duration, summary.req_per_s, summary.tok_per_s]],
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


def create_run_dir(engines: list[str]) -> Path:
    run_dir = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}"
    for engine in engines:
        (run_dir / f"results_{engine}").mkdir(parents=True)
    return run_dir


def save_report(
        run_dir: Path,
        index: int,
        title: str,
        engine: str,
        settings: dict,
        info: WorkloadInfo,
        summary: OverallMetrics,
        serve_result: ServeResult
) -> Path:
    name = f"{index:03d}_{info.kind}_rate{settings['rate']:g}_r{settings['repeat']}"
    json_path = run_dir / f"results_{engine}" / f"{name}.json"

    report = {
        "title": title,
        "name": name,
        "benchmark": "serving",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "env": environment(engine),
        "settings": {
            "engine": engine,
            "kind": info.kind,
            **settings,
            "n_short": info.n_short,
            "n_long": info.n_long,
            "mean_prompt_tokens": info.mean_tokens,
        },
        "tables": build_tables(summary, serve_result),
        "records": [
            dataclasses.asdict(timed)
            for timed in serve_result.output
        ],
    }
    with open(json_path, "w") as f:
        json.dump(report, f)

    return json_path


def load_reports(run_dir: Path) -> dict[str, dict[RunKey, dict]]:
    reports = {}
    for engine_dir in sorted(run_dir.glob("results_*")):
        engine = engine_dir.name.removeprefix("results_")
        reports[engine] = {}
        for path in sorted(engine_dir.glob("*.json")):
            with open(path) as f:
                report = json.load(f)
            settings = report["settings"]
            reports[engine][(settings["kind"], settings["rate"], settings["repeat"])] = report
    return reports


def table_by_title(report: dict, title: str) -> dict:
    return next(table for table in report["tables"] if table["title"] == title)


def comparison_values(report: dict) -> dict[str, str]:
    measured, duration, req_per_s, tok_per_s = table_by_title(report, "Throughput")["rows"][0]
    n_exact, n_requests, match_ratio = table_by_title(report, "Reference match")["rows"][0]

    values = {
        "measured": str(measured),
        "duration (s)": f"{duration:.2f}",
        "requests/s": f"{req_per_s:.2f}",
        "tokens/s": f"{tok_per_s:.1f}",
    }
    for metric, mean, p50, p99 in table_by_title(report, "Latency (ms)")["rows"]:
        values[f"{metric} mean"] = f"{mean:.2f}"
        values[f"{metric} p50"] = f"{p50:.2f}"
        values[f"{metric} p99"] = f"{p99:.2f}"

    values["exact"] = f"{n_exact}/{n_requests}"
    values["match ratio"] = f"{match_ratio:.3f}"
    return values


def comparison_table(reports: dict[str, dict[RunKey, dict]], keys: list[RunKey]) -> str:
    engines = list(reports)

    rows = []
    columns: list[str] = []
    for key in keys:
        values = {
            engine: comparison_values(reports[engine][key]) if key in reports[engine] else {}
            for engine in engines
        }
        for engine_values in values.values():
            columns = columns or list(engine_values)

        rows.append([
            *key,
            *(
                " | ".join(values[engine].get(column, "-") for engine in engines)
                for column in columns
            ),
        ])

    return render_table(["kind", "rate", "repeat", *columns], rows)


def run_info_table(reports: dict[str, dict[RunKey, dict]]) -> str:
    first = {engine: next(iter(engine_reports.values())) for engine, engine_reports in reports.items()}
    engines = list(first)

    def values(engine: str) -> dict:
        report = first[engine]
        settings = {
            key: value
            for key, value in report["settings"].items()
            if key not in RUN_SPECIFIC_SETTINGS
        }
        return {"commit": report["git"]["commit"], "dirty": report["git"]["dirty"], **report["env"], **settings}

    names = list(dict.fromkeys(name for engine in engines for name in values(engine)))
    return render_table(
        ["setting", *engines],
        [[name, *(values(engine).get(name) for engine in engines)] for name in names]
    )


def init_section(reports: dict[str, dict[RunKey, dict]]) -> str:
    return "".join(
        f"<h3>{engine}</h3>" + render_table(table["columns"], table["rows"])
        for engine, engine_reports in reports.items()
        for table in [table_by_title(next(iter(engine_reports.values())), "Init")]
    )


def run_details(report: dict) -> str:
    settings = report["settings"]
    sections = [
        render_table(
            ["setting", "value"],
            [
                [name, settings[name]]
                for name in RUN_SPECIFIC_SETTINGS
                if name in settings and name not in ("kind", "rate", "repeat")
            ]
        )
    ]
    for table in report["tables"]:
        if table["title"] not in SUMMARIZED_TABLES:
            sections.append(f"<h4>{html.escape(table['title'])}</h4>")
            sections.append(render_table(table["columns"], table["rows"]))
    return "".join(sections)


def write_run_html(run_dir: Path) -> Path:
    reports = {
        engine: engine_reports
        for engine, engine_reports in load_reports(run_dir).items()
        if engine_reports
    }
    engines = list(reports)

    keys: list[RunKey] = sorted({
        key
        for engine_reports in reports.values()
        for key in engine_reports
    })

    title = next(iter(next(iter(reports.values())).values()))["title"]

    sections = [
        "<h2>Run settings</h2>",
        run_info_table(reports),
        "<h2>Init</h2>",
        init_section(reports),
        f"<h2>Results ({' | '.join(engines)}), times in ms</h2>",
        comparison_table(reports, keys),
    ]
    for kind, rate, repeat in keys:
        columns = "".join(
            f"<div><h3>{engine}</h3>"
            + run_details(reports[engine][(kind, rate, repeat)])
            + "</div>"
            for engine in engines
            if (kind, rate, repeat) in reports[engine]
        )
        sections.append(
            f"<section><h2>{kind}, rate {rate:g}, repeat {repeat}</h2>"
            f"<div class=\"columns\">{columns}</div></section>"
        )

    html_path = run_dir / "report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_page(title, "".join(sections)))

    return html_path
