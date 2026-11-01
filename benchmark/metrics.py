import dataclasses
from collections import Counter

import numpy as np

EngineResult = tuple[list[int], float, float, float]
TimedResult = tuple[EngineResult, float]

FIRST_DIFF_BIN = 16


@dataclasses.dataclass
class RequestMetrics:
    ttft_engine: float
    ttft_overall: float
    tpot: float | None
    e2e: float
    send_lag: float


@dataclasses.dataclass
class TokenMatch:
    exact: bool
    first_diff: int
    match_ratio: float


@dataclasses.dataclass
class ReferenceMatch:
    n_exact: int
    mean_match_ratio: float
    first_diff_histogram: dict[int, int]


@dataclasses.dataclass
class OverallMetrics:
    duration: float
    req_per_s: float
    tok_per_s: float
    means: RequestMetrics
    p50: RequestMetrics
    p99: RequestMetrics
    reference_match: ReferenceMatch


def get_metrics(
        result: EngineResult,
        planned: float
) -> RequestMetrics:
    new_tokens, t_arrival, t_first, t_end = result

    ttft_engine = t_first - t_arrival
    ttft_overall = t_first - planned
    tpot = (
        (t_end - t_first) / (l - 1)
        if (l := len(new_tokens)) != 1
        else None
    )
    e2e = t_end - planned
    send_lag = t_arrival - planned

    return RequestMetrics(
        ttft_engine=ttft_engine,
        ttft_overall=ttft_overall,
        tpot=tpot,
        e2e=e2e,
        send_lag=send_lag
    )


def aggregate_metrics(
        metrics: list[RequestMetrics],
        stat
) -> RequestMetrics:
    aggregated = {}
    for field in dataclasses.fields(RequestMetrics):
        values = [
            value
            for m in metrics
            if (value := getattr(m, field.name)) is not None
        ]
        aggregated[field.name] = float(stat(values)) if values else None

    return RequestMetrics(**aggregated)


def means_metrics(
        metrics: list[RequestMetrics]
) -> RequestMetrics:
    return aggregate_metrics(metrics, np.mean)


def percentile_metrics(
        metrics: list[RequestMetrics],
        q: float
) -> RequestMetrics:
    return aggregate_metrics(metrics, lambda values: np.percentile(values, q))


def strip_eos(tokens: list[int], eos_token_ids: set[int]) -> list[int]:
    end = len(tokens)
    while end > 0 and tokens[end - 1] in eos_token_ids:
        end -= 1
    return tokens[:end]


def compare_tokens(
        tokens: list[int],
        reference: list[int],
        eos_token_ids: set[int]
) -> TokenMatch:
    tokens = strip_eos(tokens, eos_token_ids)
    reference = strip_eos(reference, eos_token_ids)

    first_diff = 0
    while (
            first_diff < min(len(tokens), len(reference))
            and tokens[first_diff] == reference[first_diff]
    ):
        first_diff += 1

    return TokenMatch(
        exact=tokens == reference,
        first_diff=first_diff,
        match_ratio=first_diff / longest if (longest := max(len(tokens), len(reference))) else 1.0
    )


def summarize_matches(matches: list[TokenMatch]) -> ReferenceMatch:
    histogram = Counter(
        match.first_diff // FIRST_DIFF_BIN * FIRST_DIFF_BIN
        for match in matches
        if not match.exact
    )

    return ReferenceMatch(
        n_exact=sum(match.exact for match in matches),
        mean_match_ratio=float(np.mean([match.match_ratio for match in matches])),
        first_diff_histogram=dict(sorted(histogram.items()))
    )


def summarize(
        output: list[TimedResult],
        references: list[list[int]],
        eos_token_ids: set[int]
) -> OverallMetrics:
    n = len(output)

    metrics = []
    start = float("inf")
    end = float("-inf")
    total_tokens = 0

    for result, planned in output:
        metrics.append(
            get_metrics(
                result=result,
                planned=planned
            )
        )

        start = min(start, planned)

        new_tokens, _, _, t_end = result
        end = max(end, t_end)

        total_tokens += len(new_tokens)

    duration = end - start
    req_per_s = n / duration
    tok_per_s = total_tokens / duration

    return OverallMetrics(
        duration=duration,
        req_per_s=req_per_s,
        tok_per_s=tok_per_s,
        means=means_metrics(metrics),
        p50=percentile_metrics(metrics, 50),
        p99=percentile_metrics(metrics, 99),
        reference_match=summarize_matches([
            compare_tokens(result[0], reference, eos_token_ids)
            for (result, _), reference in zip(output, references)
        ])
    )
