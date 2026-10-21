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

    def __init__(self, model, max_len, n_slots, batch_size=1, warmup_iters=3):
        self.model = model
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_slots = n_slots
        _slot_ids = list(range(1, n_slots))

        device = model.device

        with torch.inference_mode():
            self.static_token = torch.zeros(
                (1, batch_size), device=device, dtype=torch.long
            )
            self.static_position_ids = torch.zeros(
                (1, batch_size), device=device, dtype=torch.long
            )
            self.kv_cache = KVCacheManager(
                model.config, max_len, batch_size, n_slots, device, model.dtype
            )

            use_triton_attention(model)

            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                self.kv_cache.reset(_slot_ids)
                self.kv_cache.prepare_decode()
                self._forward()

                forward = torch.compile(
                    self._forward, fullgraph=True, dynamic=False
                )

                for _ in range(warmup_iters):
                    self.kv_cache.reset(_slot_ids)
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


def pad_slot_ids(slot_ids: list[int], batch_size: int, unused_slot: int):
    missing = batch_size - len(slot_ids)
    if missing < 0:
        raise ValueError(
            "more elements than batch size"
        )
    elif missing == 0:
        return slot_ids
    else:
        return slot_ids + [unused_slot] * missing

def run_slots(
        slot_ids: list[int],
        graph: DecodeGraph
) -> list[int]:

    if not slot_ids:
        return []

    kv_cache = graph.kv_cache
    batch_size, unused_slot = kv_cache.batch_size, kv_cache.unused_slot

    if not all(kv_cache.occupancy[slot_ids].tolist()):
        return []

    padded_slot_ids = pad_slot_ids(slot_ids, batch_size, unused_slot)
    last_position_ids = kv_cache.start[padded_slot_ids]
    last_generated = kv_cache.last_generated[padded_slot_ids]

    graph.static_position_ids.copy_(
        last_position_ids[None]
    )
    graph.static_token.copy_(
        last_generated[None]
    )
    kv_cache.write_slot_ids(slot_ids)

    logits = graph.replay()
    next_tokens = torch.argmax(logits, dim=-1)
    tokens = next_tokens[:len(slot_ids)].tolist()

    kv_cache.last_generated[slot_ids] = tokens
    return tokens













