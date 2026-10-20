import torch
import torch.cuda.nvtx as nvtx

from attn import replace_attention
from kv_cache.manager import KVCacheManager


KV_CACHE_CAPACITY = 32768


def use_triton_attention(model):
    if not getattr(model, "_uses_triton_attention", False):
        replace_attention(model)
        model._uses_triton_attention = True


class DecodeGraph:

    def __init__(self, model, max_len, batch_size=1, warmup_iters=3):
        self.model = model
        self.batch_size = batch_size
        self.max_len = max_len
        device = model.device

        with torch.inference_mode():
            self.static_token = torch.zeros(
                (1, batch_size), device=device, dtype=torch.long
            )
            self.static_position_ids = torch.zeros(
                (1, batch_size), device=device, dtype=torch.long
            )
            self.kv_cache = KVCacheManager(
                model.config, max_len, batch_size, device, model.dtype
            )

            use_triton_attention(model)

            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                self.kv_cache.reset()
                self.kv_cache.prepare_decode()
                self._forward()

                forward = torch.compile(
                    self._forward, fullgraph=True, dynamic=False
                )

                for _ in range(warmup_iters):
                    self.kv_cache.reset()
                    self.kv_cache.prepare_decode()
                    forward()
            torch.cuda.current_stream().wait_stream(side_stream)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.kv_cache.prepare_decode()
                self.static_logits = forward().logits[0]

    def _forward(self):
        outputs = self.model(
            input_ids=self.static_token,
            position_ids=self.static_position_ids,
            kv_manager=self.kv_cache,
            logits_to_keep=0,
            use_cache=False
        )
        self.kv_cache.advance()
        return outputs

    def replay(self):
        with nvtx.range("manual_graph_replay"):
            self.graph.replay()
        return self.static_logits


_decode_graphs = {}


DECODE_BATCH_BUCKETS = (1, 2, 4, 8, 16)


def decode_bucket(batch_size):
    for bucket in DECODE_BATCH_BUCKETS:
        if batch_size <= bucket:
            return bucket
    raise ValueError(
        f"batch size {batch_size} en büyük kovadan ({DECODE_BATCH_BUCKETS[-1]}) büyük"
    )


def get_decode_graph(model, batch_size=1):
    key = (model, batch_size)
    if key not in _decode_graphs:
        _decode_graphs[key] = DecodeGraph(
            model, KV_CACHE_CAPACITY // batch_size, batch_size
        )
    return _decode_graphs[key]


COMPILE_PREFILL = True


def prefill_forward(model, input_ids, position_ids, kv_cache):
    hidden = model.model(
        input_ids=input_ids,
        position_ids=position_ids,
        kv_manager=kv_cache,
        use_cache=False
    ).last_hidden_state
    return model.lm_head(hidden[0, kv_cache.cu_seqlens[1:] - 1])


_prefill_fns = {}


def get_prefill_fn(model):
    if model not in _prefill_fns:
        _prefill_fns[model] = torch.compile(prefill_forward, fullgraph=True)
    return _prefill_fns[model]


def generate_batch(model, prompts, eos_token_ids, max_new_tokens=10, on_step=None):
    B = len(prompts)
    if B == 0 or max_new_tokens <= 0:
        return [[] for _ in prompts]
    if any(len(p) == 0 for p in prompts):
        raise ValueError("boş prompt desteklenmiyor")

    bucket = decode_bucket(B)
    prompts = list(prompts) + [[0]] * (bucket - B)
    lens_list = [len(p) for p in prompts]
    T = max(lens_list)

    with torch.inference_mode():
        use_triton_attention(model)
        decode_graph = get_decode_graph(model, bucket)
        kv_cache = decode_graph.kv_cache
        device = model.device

        if T + max_new_tokens > decode_graph.max_len:
            raise ValueError(
                f"prompt + yeni token ({T} + {max_new_tokens}) dizi başına KV cache "
                f"kapasitesini ({decode_graph.max_len}, batch {bucket}) aşıyor"
            )

        input_ids = torch.tensor(
            [t for p in prompts for t in p], device=device
        )[None]
        total_T = input_ids.shape[1]

        kv_cache.reset()
        kv_cache.prepare_prefill(lens_list)
        position_ids = kv_cache.pos[None]

        prefill = get_prefill_fn(model) if COMPILE_PREFILL else prefill_forward
        if COMPILE_PREFILL and total_T > 1:
            torch._dynamo.mark_dynamic(input_ids, 1)
            torch._dynamo.mark_dynamic(position_ids, 1)
            torch._dynamo.mark_dynamic(kv_cache.pos, 0)
            torch._dynamo.mark_dynamic(kv_cache.seq_idx, 0)
        logits = prefill(model, input_ids, position_ids, kv_cache)
        kv_cache.advance()

        outputs = [[] for _ in range(B)]
        finished = [False] * B

        with nvtx.range("manual_batch_decode"):
            for step in range(max_new_tokens):
                next_tokens = torch.argmax(logits, dim=-1)

                tokens = next_tokens[:B].tolist()

                if on_step is not None:
                    on_step()

                for b, token in enumerate(tokens):
                    if finished[b]:
                        continue
                    if token in eos_token_ids:
                        finished[b] = True
                    else:
                        outputs[b].append(token)

                if all(finished) or step == max_new_tokens - 1:
                    break

                decode_graph.static_token.copy_(next_tokens[None])
                decode_graph.static_position_ids.copy_(kv_cache.start[None])

                with nvtx.range("decode_forward"):
                    logits = decode_graph.replay()

    return outputs


def generate_manual(model, inputs, eos_token_ids, max_new_tokens=10, on_token=None):
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("sadece batch size 1")
    if not inputs["attention_mask"].bool().all():
        raise ValueError("prompt'ta pad desteklenmiyor")
    return generate_batch(
        model, [inputs["input_ids"][0].tolist()], eos_token_ids, max_new_tokens,
        on_step=on_token
    )[0]
