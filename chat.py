import argparse
from threading import Thread

import torch
from transformers import AutoTokenizer

from benchmark.serve import load_eos_token_ids
from common import MODEL_ID
from scheduler.scheduler import Scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the custom inference engine.")
    parser.add_argument("--fp8", action="store_true", help="Use FP8 projections.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    print("Loading the model and preparing CUDA graphs...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    scheduler = Scheduler(
        model_id=MODEL_ID,
        eos_token_ids=load_eos_token_ids(MODEL_ID, tokenizer),
        max_new_tokens=args.max_new_tokens,
        n_slots=2,
        page_size=16,
        token_budget=Scheduler.FULL_GRAPH_TOKEN_THRESHOLD + Scheduler.PIECEWISE_TOKEN_STEP,
        dtype=torch.float16,
        quantization="fp8" if args.fp8 else None,
    )
    worker = Thread(target=scheduler.loop, daemon=True)
    worker.start()

    messages: list[dict[str, str]] = []
    print("Ready. Use /reset to clear the conversation or /exit to quit.")
    try:
        while True:
            message = input("You> ").strip()
            if not message:
                continue
            if message == "/exit":
                break
            if message == "/reset":
                messages.clear()
                print("Conversation cleared.")
                continue

            turn = messages + [{"role": "user", "content": message}]
            token_ids = tokenizer.apply_chat_template(
                turn, tokenize=True, add_generation_prompt=True, return_dict=True
            )["input_ids"]
            try:
                result = scheduler.call(token_ids).result()
            except ValueError as exc:
                print(f"Prompt rejected: {exc} Use /reset to clear the conversation.")
                continue

            reply = tokenizer.decode(result.new_tokens, skip_special_tokens=True)
            print(f"Assistant> {reply}")
            messages = turn + [{"role": "assistant", "content": reply}]
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        scheduler.stop()
        worker.join()


if __name__ == "__main__":
    main()
