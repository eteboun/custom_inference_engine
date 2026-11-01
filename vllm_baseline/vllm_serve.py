import dataclasses
import time
from collections import deque

from vllm import LLMEngine, SamplingParams, TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind

from benchmark.metrics import TimedResult


@dataclasses.dataclass
class PendingRequest:
    planned: float
    t_arrival: float = 0.0
    t_first: float = 0.0
    t_end: float = 0.0
    new_tokens: list[int] = dataclasses.field(default_factory=list)

    def result(self) -> TimedResult:
        return (self.new_tokens, self.t_arrival, self.t_first, self.t_end), self.planned


def submit_due(
        engine: LLMEngine,
        requests: list[list[int]],
        pending: list[PendingRequest],
        waiting: deque[int],
        params: SamplingParams,
        tag: str
) -> None:
    while waiting and pending[waiting[0]].planned <= time.perf_counter():
        i = waiting.popleft()
        pending[i].t_arrival = time.perf_counter()
        engine.add_request(
            f"{tag}-{i}",
            TokensPrompt(prompt_token_ids=requests[i]),
            params
        )


def record_outputs(
        outputs: list[RequestOutput],
        pending: list[PendingRequest]
) -> None:
    now = time.perf_counter()
    for output in outputs:
        request = pending[int(output.request_id.rsplit("-", 1)[1])]
        new_tokens = list(output.outputs[0].token_ids)

        if new_tokens and not request.new_tokens:
            request.t_first = now

        request.new_tokens.extend(new_tokens)

        if output.finished:
            request.t_end = now


def serve_vllm(
        engine: LLMEngine,
        requests: list[list[int]],
        arrivals: list[float],
        eos_token_ids: set[int],
        max_new_tokens: int,
        tag: str
) -> list[TimedResult]:

    params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        stop_token_ids=list(eos_token_ids),
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )

    t0 = time.perf_counter()
    pending = [PendingRequest(planned=t0 + arrival) for arrival in arrivals]
    waiting = deque(range(len(requests)))

    while waiting or engine.has_unfinished_requests():
        submit_due(engine, requests, pending, waiting, params, tag)

        if not engine.has_unfinished_requests():
            time.sleep(max(0., pending[waiting[0]].planned - time.perf_counter()))
            continue

        record_outputs(engine.step(), pending)

    return [request.result() for request in pending]
