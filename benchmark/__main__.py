import argparse

from transformers import AutoTokenizer, GenerationConfig

from benchmark.metrics import summarize
from benchmark.reference import load_references
from benchmark.report import save_report
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--engine", choices=["ours", "vllm"], required=True)
    parser.add_argument("--kinds", nargs="+", choices=["short", "long", "mixed"], default=["short", "long", "mixed"])
    parser.add_argument("--rates", type=float, nargs="+", default=[4.0, 8.0, float("inf")])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-requests", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--n-slots", type=int, default=33)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=1024)
    args = parser.parse_args()

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
        if args.engine == "ours"
        else VllmServer(args, eos_token_ids)
    )

    settings = {
        "n_requests": args.n_requests,
        "max_new_tokens": args.max_new_tokens,
        "n_slots": args.n_slots,
        "page_size": args.page_size,
        "token_budget": args.token_budget,
        "prompt_seed": args.seed,
    }

    for repeat in range(args.repeats):
        for kind, (requests, info, references) in workloads.items():
            for rate in args.rates:
                arrival_seed = args.seed + repeat

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
                    eos_token_ids=eos_token_ids
                )

                json_path, _ = save_report(
                    title=args.title,
                    engine=args.engine,
                    settings={
                        **settings,
                        "rate": rate,
                        "repeat": repeat,
                        "arrival_seed": arrival_seed,
                    },
                    info=info,
                    summary=summary,
                    serve_result=serve_result
                )
                print(
                    f"{kind} rate={rate:g} repeat={repeat} "
                    f"tok/s={summary.tok_per_s:.1f} "
                    f"exact={summary.reference_match.n_exact}/{args.n_requests} "
                    f"saved={json_path.name}",
                    flush=True
                )

    server.close()


if __name__ == "__main__":
    main()
