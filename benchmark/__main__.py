import argparse
import threading

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from common import MODEL_ID
from benchmark.metrics import compute_metrics, table_header, table_row
from benchmark.report import ARCHITECTURE, save_results
from benchmark.scheduler_client import serve_scheduler
from benchmark.workload import build_prompts, load_eos_token_ids, poisson_arrivals


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["ours", "vllm"], required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[2, 4, 8, float("inf")])
    parser.add_argument("--n-requests", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--n-slots", type=int, default=5)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def make_ours_server(args, eos_token_ids):
    from scheduler.scheduler import Scheduler

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to("cuda").eval()
    sched = Scheduler(model, eos_token_ids, args.max_new_tokens, args.n_slots, args.page_size,
                      args.token_budget, torch.float16)
    thread = threading.Thread(target=sched.loop, daemon=True)
    thread.start()

    def serve(prompts, arrivals, tag):
        return serve_scheduler(sched, prompts, arrivals)

    def close():
        sched.stop()
        thread.join()

    return serve, close


def make_vllm_server(args, eos_token_ids):
    from vllm_baseline.vllm_generate import build_vllm_engine
    from vllm_baseline.vllm_serve import serve_vllm

    overrides = {}
    if args.max_num_seqs is not None:
        overrides["max_num_seqs"] = args.max_num_seqs
    engine = build_vllm_engine(**overrides)

    def serve(prompts, arrivals, tag):
        return serve_vllm(engine, prompts, arrivals, eos_token_ids, args.max_new_tokens, tag)

    return serve, lambda: None


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    eos_token_ids = load_eos_token_ids(tokenizer)
    prompts = build_prompts(tokenizer, args.n_requests, args.seed)

    make_server = make_ours_server if args.impl == "ours" else make_vllm_server
    serve, close = make_server(args, eos_token_ids)

    warmup = prompts[:args.warmup_requests]
    serve(warmup, [0.0] * len(warmup), "warmup")

    title = args.title or (ARCHITECTURE if args.impl == "ours" else "vLLM baseline")
    lines = [
        title,
        f"impl={args.impl}  n_requests={args.n_requests}  max_new_tokens={args.max_new_tokens}  "
        f"n_slots={args.n_slots}  max_num_seqs={args.max_num_seqs}  seed={args.seed}  (ms)",
        table_header(),
    ]
    print("\n".join(lines))

    runs = []
    for rate in args.rates:
        arrivals = poisson_arrivals(args.n_requests, rate, args.seed)
        records = serve(prompts, arrivals, f"rate{rate:g}")
        metrics = compute_metrics(records)
        lines.append(table_row(rate, metrics))
        print(lines[-1])
        runs.append({"rate": rate, "metrics": metrics, "records": records})

    close()

    path = save_results(title, args.impl, vars(args), "\n".join(lines), runs)
    print(f"saved: {path}")
