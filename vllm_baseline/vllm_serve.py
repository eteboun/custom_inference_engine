import asyncio
import time

from vllm import AsyncEngineArgs, SamplingParams, TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from benchmark.metrics import TimedResult
from scheduler.models import RequestResult
from vllm_baseline.vllm_generate import vllm_engine_options


def build_async_vllm_engine(**overrides) -> AsyncLLM:
    args = AsyncEngineArgs(**vllm_engine_options(**overrides))
    return AsyncLLM.from_engine_args(args)


async def send_request(
    engine: AsyncLLM,
    prompt: TokensPrompt,
    params: SamplingParams,
    request_id: str,
    planned: float,
) -> TimedResult:
    await asyncio.sleep(max(0.0, planned - time.perf_counter()))
    arrival = time.perf_counter()
    first_token = None
    finished_at = None
    tokens: list[int] = []
    async for output in engine.generate(prompt, params, request_id):
        now = time.perf_counter()
        new_tokens = output.outputs[0].token_ids if output.outputs else ()
        if new_tokens and first_token is None:
            first_token = now
        tokens.extend(new_tokens)

        if output.finished:
            finished_at = now

    if finished_at is None:
        raise RuntimeError(f"vLLM stream ended before request {request_id} finished.")
    return TimedResult(
        result=RequestResult(
            new_tokens=tokens,
            t_arrival=arrival,
            t_first=first_token if first_token is not None else finished_at,
            t_end=finished_at,
        ),
        planned=planned,
    )


async def serve_vllm(
    engine: AsyncLLM,
    requests: list[list[int]],
    arrivals: list[float],
    eos_token_ids: set[int],
    max_new_tokens: int,
    tag: str,
) -> list[TimedResult]:
    params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        stop_token_ids=list(eos_token_ids),
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )
    prompts = [TokensPrompt(prompt_token_ids=request) for request in requests]
    start = time.perf_counter()
    return await asyncio.gather(*(
        send_request(engine, prompt, params, f"{tag}-{index}", start + arrival)
        for index, (prompt, arrival) in enumerate(zip(prompts, arrivals, strict=True))
    ))
