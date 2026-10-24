import time


def serve_scheduler(sched, prompts, arrivals):
    futs = []
    t0 = time.perf_counter()
    for prompt, arrival in zip(prompts, arrivals):
        delay = t0 + arrival - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        futs.append(sched.call([prompt])[0])

    records = []
    for prompt, arrival, fut in zip(prompts, arrivals, futs):
        tokens, _, t_first, t_end = fut.result()
        records.append({
            "prompt_len": len(prompt),
            "tokens": tokens,
            "planned": t0 + arrival,
            "t_first": t_first,
            "t_end": t_end,
        })
    return records
