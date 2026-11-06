import argparse
import dataclasses
import json
import math
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig

from common import MODEL_ID
from feature_bench.common import PAGE_SIZE, save_report
from scheduler.scheduler import Scheduler

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATASET = Path(__file__).resolve().parents[1] / "datasets" / "gsm8k" / "test.jsonl"
VARIANTS = ["fp16", "fp8"]
INIT_COMPONENTS = ["model", "kv_cache", "piecewise", "full"]
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


def run_worker(variant: str, args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eos = GenerationConfig.from_pretrained(MODEL_ID).eos_token_id
    eos_token_ids = set(eos) if isinstance(eos, list) else {eos}

    questions = load_questions(args.limit)
    prompts = build_prompts(tokenizer, questions)

    engine = Scheduler(
        model_id=MODEL_ID,
        eos_token_ids=eos_token_ids,
        max_new_tokens=args.max_new_tokens,
        n_slots=args.n_slots,
        page_size=PAGE_SIZE,
        token_budget=args.token_budget,
        dtype=torch.float16,
        quantization="fp8" if variant == "fp8" else None
    )
    thread = threading.Thread(target=engine.loop, daemon=True)
    thread.start()

    start = time.perf_counter()
    futures = [engine.call(prompt) for prompt in prompts]
    results = [future.result() for future in futures]
    seconds = time.perf_counter() - start

    engine.stop()
    thread.join()

    answers = []
    for question, result in zip(questions, results):
        tokens = result.new_tokens
        prediction = extract_answer(tokenizer.decode(tokens, skip_special_tokens=True))
        reference = reference_answer(question["answer"])
        answers.append({
            "reference": reference,
            "prediction": prediction,
            "correct": prediction is not None and math.isclose(prediction, reference, abs_tol=1e-6),
            "tokens": len(tokens),
            "truncated": tokens[-1] not in eos_token_ids,
        })

    with open(args.worker_out, "a") as f:
        f.write(json.dumps({
            "variant": variant,
            "calibration_seconds": engine.tracker.init.calibration_seconds,
            "init_seconds": engine.tracker.init.total_seconds,
            "init_components": {
                name: dataclasses.asdict(stats)
                for name in INIT_COMPONENTS
                if (stats := getattr(engine.tracker.init, name)) is not None
            },
            "generation_seconds": seconds,
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
        run["variant"], len(answers),
        sum(a["correct"] for a in answers),
        sum(a["correct"] for a in answers) / len(answers),
        sum(a["prediction"] is None for a in answers),
        sum(a["truncated"] for a in answers),
        tokens / len(answers),
        run["generation_seconds"],
        tokens / run["generation_seconds"],
        run["init_seconds"],
        run["calibration_seconds"],
    ]


def build_tables(runs: dict[str, dict]) -> list[dict]:
    fp16 = runs["fp16"]["answers"]
    fp8 = runs["fp8"]["answers"]
    pairs = list(zip(fp16, fp8))
    fp16_only = sum(a["correct"] and not b["correct"] for a, b in pairs)
    fp8_only = sum(b["correct"] and not a["correct"] for a, b in pairs)

    return [
        {
            "title": "Accuracy",
            "columns": [
                "variant", "questions", "correct", "accuracy", "no answer found",
                "hit max_new_tokens", "mean generated tokens", "generation seconds",
                "generated tokens/s", "scheduler init seconds", "calibration seconds",
            ],
            "rows": [accuracy_row(runs[variant]) for variant in VARIANTS],
        },
        {
            "title": "Scheduler init",
            "columns": [
                "variant", "component", "seconds", "memory MiB",
                "allocated after MiB", "reserved after MiB",
            ],
            "rows": [
                [
                    variant, name, stats["seconds"], stats["memory"] / 2 ** 20,
                    stats["memory_allocated"] / 2 ** 20, stats["memory_reserved"] / 2 ** 20,
                ]
                for variant in VARIANTS
                for name, stats in runs[variant]["init_components"].items()
            ] + [
                [variant, "total", runs[variant]["init_seconds"], None, None, None]
                for variant in VARIANTS
            ],
        },
        {
            "title": "Paired by question",
            "columns": ["both correct", "fp16 only", "fp8 only", "both wrong", "McNemar exact p"],
            "rows": [[
                sum(a["correct"] and b["correct"] for a, b in pairs),
                fp16_only,
                fp8_only,
                sum(not a["correct"] and not b["correct"] for a, b in pairs),
                mcnemar_p(fp16_only, fp8_only),
            ]],
        },
        {
            "title": "Questions where only one variant is correct",
            "columns": [
                "index", "reference", "fp16 prediction", "fp8 prediction",
                "fp16 tokens", "fp8 tokens",
            ],
            "rows": [
                [i, a["reference"], a["prediction"], b["prediction"], a["tokens"], b["tokens"]]
                for i, (a, b) in enumerate(pairs)
                if a["correct"] != b["correct"]
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--n-slots", type=int, default=33)
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--worker", choices=VARIANTS, default=None)
    parser.add_argument("--worker-out", type=Path, default=None)
    args = parser.parse_args()

    if args.worker is not None:
        run_worker(args.worker, args)
        return

    if args.title is None:
        parser.error("--title is required")

    worker_args = [
        "--max-new-tokens", str(args.max_new_tokens),
        "--n-slots", str(args.n_slots),
        "--token-budget", str(args.token_budget),
    ]
    if args.limit is not None:
        worker_args += ["--limit", str(args.limit)]

    with tempfile.TemporaryDirectory() as tmp:
        worker_out = Path(tmp) / "runs.jsonl"
        for variant in VARIANTS:
            subprocess.run(
                [
                    sys.executable, "-m", "feature_bench.fp8.gsm8k",
                    "--worker", variant,
                    "--worker-out", str(worker_out),
                    *worker_args,
                ],
                check=True,
            )

        with open(worker_out) as f:
            runs = {run["variant"]: run for run in map(json.loads, f)}

    path = save_report(
        name="gsm8k",
        title=args.title,
        settings={
            "dataset": "GSM8K test (openai/grade-school-math)",
            "variants": VARIANTS,
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
        results_dir=args.results_dir
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
