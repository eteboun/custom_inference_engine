import argparse
import itertools
import math
import random
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig

from benchmark.metrics import summarize
from benchmark.reference import load_references
from benchmark.report import create_run_dir, save_report, write_run_html
from benchmark.serve import OursServer, Server, VllmServer
from benchmark.workload import WorkloadInfo, build_workload, poisson_arrivals
from common import MODEL_ID


def load_eos_token_ids(tokenizer) -> set[int]:
    eos = GenerationConfig.from_pretrained(MODEL_ID).eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    return set(eos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--engines", nargs="+", choices=["ours", "vllm"], default=["ours", "vllm"])
    parser.add_argument("--kinds", nargs="+", choices=["short", "long", "mixed"], default=["short", "long", "mixed"])
    parser.add_argument("--rates", type=float, nargs="+", default=[4.0, 8.0, float("inf")])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=16)
    parser.add_argument("--trim", type=float, default=0.05)
    parser.add_argument("--n-requests", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--n-slots", type=int, default=33)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--worker-engine", choices=["ours", "vllm"], default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def run_engine(args: argparse.Namespace) -> None:
    engine: str = args.worker_engine
    run_dir: Path = args.run_dir

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eos_token_ids = load_eos_token_ids(tokenizer)

    workloads: dict[str, tuple[list[list[int]], WorkloadInfo, list[list[int]]]] = {}
    for kind in args.kinds:
        requests, info = build_workload(
            tokenizer=tokenizer,
            kind=kind,
            n=args.n_requests,
            seed=args.seed
        )
        references = load_references(
            requests=requests,
            eos_token_ids=eos_token_ids,
            max_new_tokens=args.max_new_tokens
        )
        workloads[kind] = requests, info, references

    server: Server = (
        OursServer(args, eos_token_ids)
        if engine == "ours"
        else VllmServer(args, eos_token_ids)
    )

    warmup_requests, _ = build_workload(
        tokenizer=tokenizer,
        kind="mixed",
        n=args.warmup_requests,
        seed=args.seed
    )
    server.serve(
        requests=warmup_requests,
        arrivals=[0.0] * args.warmup_requests
    )

    settings = {
        "n_requests": args.n_requests,
        "max_new_tokens": args.max_new_tokens,
        "n_slots": args.n_slots,
        "page_size": args.page_size,
        "token_budget": args.token_budget,
        "prompt_seed": args.seed,
    }

    configs = list(itertools.product(range(args.repeats), args.kinds, args.rates))
    random.Random(args.seed).shuffle(configs)

    for index, (repeat, kind, rate) in enumerate(configs):
        requests, info, references = workloads[kind]

        arrival_seed = args.seed + repeat
        trim = 0.0 if math.isinf(rate) else args.trim

        serve_result = server.serve(
            requests=requests,
            arrivals=poisson_arrivals(
                n=args.n_requests,
                seed=arrival_seed,
                rate=rate
            )
        )

        summary = summarize(
            output=serve_result.output,
            references=references,
            eos_token_ids=eos_token_ids,
            trim=trim
        )

        save_report(
            run_dir=run_dir,
            index=index,
            title=args.title,
            engine=engine,
            settings={
                **settings,
                "rate": rate,
                "repeat": repeat,
                "arrival_seed": arrival_seed,
                "trim": trim,
            },
            info=info,
            summary=summary,
            serve_result=serve_result
        )
        write_run_html(run_dir)

        print(
            f"{engine} {kind} rate={rate:g} repeat={repeat} "
            f"tok/s={summary.tok_per_s:.1f} "
            f"exact={summary.reference_match.n_exact}/{args.n_requests}",
            flush=True
        )

    server.close()


def main() -> None:
    args = parse_args()

    if args.worker_engine is not None:
        run_engine(args)
        return

    run_dir = create_run_dir(args.engines)
    for engine in args.engines:
        subprocess.run(
            [
                sys.executable, "-m", "benchmark", *sys.argv[1:],
                "--worker-engine", engine,
                "--run-dir", str(run_dir),
            ],
            check=True,
        )

    print(f"report: {run_dir / 'report.html'}")


if __name__ == "__main__":
    main()
