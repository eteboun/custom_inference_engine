import argparse
import dataclasses
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from transformers import AutoTokenizer

from benchmark.serve import ENGINES, QUANTIZATION, build_server, load_eos_token_ids
from common import MODEL_ID
from feature_bench.common import PAGE_SIZE, save_report
from scheduler.models import ComponentInitStats

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATASET = Path(__file__).resolve().parents[1] / "datasets" / "gsm8k" / "test.jsonl"
PAIRS = [("ours", "ours-fp8"), ("vllm", "vllm-fp8"), ("ours", "vllm"), ("ours-fp8", "vllm-fp8")]
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def load_questions(limit: int | None) -> list[dict]:
    with open(DATASET, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit]


def parse_number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def reference_answer(answer: str) -> float:
    return parse_number(answer.split("####")[-1].strip())


def extract_answer(text: str) -> float | None:
    boxed = text.rfind("\\boxed{")
    numbers = NUMBER.findall(text[boxed:]) if boxed != -1 else []
    if not numbers:
        numbers = NUMBER.findall(text)[-1:]
    return parse_number(numbers[0]) if numbers else None


def build_prompts(tokenizer, questions: list[dict]) -> list[list[int]]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{q['question']}\n{INSTRUCTION}"}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        for q in questions
    ]


def run_worker(engine: str, args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos_token_ids = load_eos_token_ids(args.model, tokenizer)

    questions = load_questions(args.limit)
    prompts = build_prompts(tokenizer, questions)

    server = build_server(
        engine,
        argparse.Namespace(
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            n_slots=args.n_slots,
            page_size=PAGE_SIZE,
            token_budget=args.token_budget,
        ),
        eos_token_ids
    )

    start = time.perf_counter()
    result = server.serve(requests=prompts, arrivals=[0.0] * len(prompts))
    seconds = time.perf_counter() - start
    server.close()

    answers = []
    for question, timed in zip(questions, result.output):
        tokens = timed.result.new_tokens
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        prediction = extract_answer(text)
        reference = reference_answer(question["answer"])
        answers.append({
            "reference": reference,
            "prediction": prediction,
            "correct": prediction is not None and math.isclose(prediction, reference, abs_tol=1e-6),
            "tokens": len(tokens),
            "truncated": len(tokens) >= args.max_new_tokens and tokens[-1] not in eos_token_ids,
            "text": text,
        })

    init = result.init
    with open(args.worker_out, "a") as f:
        f.write(json.dumps({
            "engine": engine,
            "calibration_seconds": init.calibration_seconds,
            "init_seconds": init.total_seconds,
            "init_components": {
                field.name: dataclasses.asdict(stats)
                for field in dataclasses.fields(init)
                if isinstance(stats := getattr(init, field.name), ComponentInitStats)
            },
            "generation_seconds": seconds,
            "preemptions": None if result.run is None else result.run.preemptions,
            "answers": answers,
        }) + "\n")


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def accuracy_row(run: dict) -> list:
    answers = run["answers"]
    tokens = sum(a["tokens"] for a in answers)
    return [
        run["engine"], len(answers),
        sum(a["correct"] for a in answers),
        sum(a["correct"] for a in answers) / len(answers),
        sum(a["prediction"] is None for a in answers),
        sum(a["truncated"] for a in answers),
        sum(a["truncated"] and a["correct"] for a in answers),
        tokens / len(answers),
        run["generation_seconds"],
        tokens / run["generation_seconds"],
        run["init_seconds"],
        run["calibration_seconds"],
        run["preemptions"],
    ]


def paired_row(first: str, second: str, subset: str, pairs: list[tuple[dict, dict]]) -> list:
    first_only = sum(a["correct"] and not b["correct"] for a, b in pairs)
    second_only = sum(b["correct"] and not a["correct"] for a, b in pairs)
    return [
        first, second, subset, len(pairs),
        sum(a["correct"] and b["correct"] for a, b in pairs),
        first_only,
        second_only,
        sum(not a["correct"] and not b["correct"] for a, b in pairs),
        mcnemar_p(first_only, second_only),
    ]


def build_tables(runs: dict[str, dict]) -> list[dict]:
    pairs = [(first, second) for first, second in PAIRS if first in runs and second in runs]

    paired_rows = []
    discordant_tables = []
    for first, second in pairs:
        both = list(zip(runs[first]["answers"], runs[second]["answers"]))
        complete = [(a, b) for a, b in both if not a["truncated"] and not b["truncated"]]
        paired_rows.append(paired_row(first, second, "all", both))
        paired_rows.append(paired_row(first, second, "neither hit max_new_tokens", complete))
        discordant_tables.append({
            "title": f"Questions where only one of {first} and {second} is correct",
            "columns": [
                "index", "reference", f"{first} prediction", f"{second} prediction",
                f"{first} tokens", f"{second} tokens",
                f"{first} hit max_new_tokens", f"{second} hit max_new_tokens",
            ],
            "rows": [
                [
                    i, a["reference"], a["prediction"], b["prediction"], a["tokens"], b["tokens"],
                    a["truncated"], b["truncated"],
                ]
                for i, (a, b) in enumerate(both)
                if a["correct"] != b["correct"]
            ],
        })

    return [
        {
            "title": "Accuracy",
            "columns": [
                "engine", "questions", "correct", "accuracy", "no answer found",
                "hit max_new_tokens", "correct among hit max_new_tokens", "mean generated tokens",
                "generation seconds", "generated tokens/s", "init seconds",
                "calibration seconds", "preemptions",
            ],
            "rows": [accuracy_row(run) for run in runs.values()],
        },
        {
            "title": "Init components",
            "columns": [
                "engine", "component", "seconds", "memory MiB",
                "allocated after MiB", "reserved after MiB",
            ],
            "rows": [
                [
                    engine, name, stats["seconds"], stats["memory"] / 2 ** 20,
                    stats["memory_allocated"] / 2 ** 20, stats["memory_reserved"] / 2 ** 20,
                ]
                for engine, run in runs.items()
                for name, stats in run["init_components"].items()
            ],
        },
        {
            "title": "Paired by question",
            "columns": [
                "first", "second", "questions", "count", "both correct", "first only",
                "second only", "both wrong", "McNemar exact p",
            ],
            "rows": paired_rows,
        },
        *discordant_tables,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default=None)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=ENGINES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--n-slots", type=int, default=33)
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--worker", choices=ENGINES, default=None)
    parser.add_argument("--worker-out", type=Path, default=None)
    args = parser.parse_args()

    if args.worker is not None:
        run_worker(args.worker, args)
        return

    if args.title is None:
        parser.error("--title is required")

    worker_args = [
        "--model", args.model,
        "--max-new-tokens", str(args.max_new_tokens),
        "--n-slots", str(args.n_slots),
        "--token-budget", str(args.token_budget),
    ]
    if args.limit is not None:
        worker_args += ["--limit", str(args.limit)]

    with tempfile.TemporaryDirectory() as tmp:
        worker_out = Path(tmp) / "runs.jsonl"
        for engine in args.engines:
            subprocess.run(
                [
                    sys.executable, "-m", "feature_bench.fp8.gsm8k",
                    "--worker", engine,
                    "--worker-out", str(worker_out),
                    *worker_args,
                ],
                check=True,
            )

        with open(worker_out) as f:
            runs = {run["engine"]: run for run in map(json.loads, f)}

    path = save_report(
        name="gsm8k",
        title=args.title,
        settings={
            "dataset": "GSM8K test (openai/grade-school-math)",
            "model": args.model,
            "engines": args.engines,
            "quantization": {engine: QUANTIZATION.get(engine) for engine in args.engines},
            "instruction": INSTRUCTION,
            "answer": "first number after the last \\boxed{, else the last number in the output",
            "decoding": "greedy",
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "n_slots": args.n_slots,
            "token_budget": args.token_budget,
            "page_size": PAGE_SIZE,
        },
        tables=build_tables(runs),
        results_dir=args.results_dir,
        records=[
            {
                "index": i,
                "question": question["question"],
                "reference": question["answer"],
                **{engine: runs[engine]["answers"][i] for engine in runs},
            }
            for i, question in enumerate(load_questions(args.limit))
        ]
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
