import atexit
import itertools
import os

from common import MODEL_ID


VLLM_GPU_MEMORY_UTILIZATION = 0.4


_vllm_engine = None


_vllm_request_ids = itertools.count()


def build_vllm_engine(**overrides):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLMEngine, EngineArgs

    engine = LLMEngine.from_engine_args(EngineArgs(**{
        "model": MODEL_ID,
        "dtype": "float16",
        "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "enable_prefix_caching": False,
        **overrides,
    }))
    atexit.register(engine.engine_core.shutdown)
    return engine


def get_vllm_engine():
    global _vllm_engine
    if _vllm_engine is None:
        _vllm_engine = build_vllm_engine()
    return _vllm_engine


def generate_vllm(model, inputs, eos_token_ids, max_new_tokens=10, on_token=None):
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    if max_new_tokens <= 0:
        return []

    engine = get_vllm_engine()

    params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        stop_token_ids=list(eos_token_ids),
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )
    engine.add_request(
        f"req{next(_vllm_request_ids)}",
        TokensPrompt(prompt_token_ids=inputs["input_ids"][0].tolist()),
        params,
    )

    new_ids = []
    while engine.has_unfinished_requests():
        for output in engine.step():
            token_ids = output.outputs[0].token_ids
            new_ids.extend(token_ids)
            if on_token is not None:
                for _ in token_ids:
                    on_token()

    if new_ids and new_ids[-1] in eos_token_ids:
        new_ids = new_ids[:-1]

    return new_ids
