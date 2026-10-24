import time
from collections import deque


def serve_vllm(engine, prompts, arrivals, eos_token_ids, max_new_tokens, tag):
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        stop_token_ids=list(eos_token_ids),
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )

    n = len(prompts)
    tokens = [[] for _ in range(n)]
    t_first = [None] * n
    t_end = [None] * n
    index = {}
    pending = deque(range(n))

    t0 = time.perf_counter()
    while pending or engine.has_unfinished_requests():
        now = time.perf_counter() - t0
        while pending and arrivals[pending[0]] <= now:
            i = pending.popleft()
            request_id = f"{tag}-{i}"
            index[request_id] = i
            engine.add_request(
                request_id, TokensPrompt(prompt_token_ids=prompts[i]), params
            )

        if not engine.has_unfinished_requests():
            time.sleep(max(0.0, t0 + arrivals[pending[0]] - time.perf_counter()))
            continue

        outputs = engine.step()
        t = time.perf_counter()
        for output in outputs:
            i = index[output.request_id]
            new = output.outputs[0].token_ids
            if new and t_first[i] is None:
                t_first[i] = t
            tokens[i].extend(new)
            if output.finished:
                t_end[i] = t

    return [
        {
            "prompt_len": len(prompts[i]),
            "tokens": tokens[i],
            "planned": t0 + arrivals[i],
            "t_first": t_first[i],
            "t_end": t_end[i],
        }
        for i in range(n)
    ]
