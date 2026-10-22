import torch
import torch.cuda.nvtx as nvtx

from attn import replace_attention
from kv_cache.manager import KVCacheManager
from typing import ClassVar

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


class Scheduler:

    DECODE_BATCH_BUCKETS: ClassVar[tuple] = (1, 2, 4, 8, 16)
    KV_CACHE_CAPACITY: ClassVar[int] = 32768

    def __init__(
            self,
            model,
            eos_id: int,
            max_new_tokens: int,
            n_slots: int
    ) -> None:
        if max_new_tokens > self.KV_CACHE_CAPACITY:
            raise ValueError(
                f"max_new_tokens can't be larger than f{self.KV_CACHE_CAPACITY}."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive."
            )

        if n_slots <= 0:
            raise ValueError(
                f"n_slots must be positive."
            )

        try:
            use_triton_attention(model)

        except Exception as exc:
            raise RuntimeError(
                "Failed to switch triton attention:"
                f" {exc}"
            )

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.n_slots = n_slots
        self.eos_id = eos_id

        self.decode_graphs = {}

    def _decode_bucket(self, batch_size: int):
        for bucket in self.DECODE_BATCH_BUCKETS:
            if batch_size <= bucket:
                return bucket
        raise ValueError(
            f"batch size {batch_size} is greater than the largest bucket {self.DECODE_BATCH_BUCKETS[-1]}"
        )

    def _get_decode_graph(self, batch_size=1):
        if batch_size not in self.decode_graphs:
            self.decode_graphs[batch_size] = DecodeGraph(
                self.model, self.KV_CACHE_CAPACITY // batch_size, batch_size
            )
        return self.decode_graphs[batch_size]

    @torch.compile(fullgraph=True)
    def _prefill_forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache: KVCacheManager
    ) -> torch.Tensor:
        hidden = self.model.model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_manager=kv_cache,
            use_cache=False
        ).last_hidden_state
        return self.model.lm_head(hidden[0, kv_cache.cu_seqlens[1:] - 1])

    @staticmethod
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

    def init_slots(
            self,
            prompts: list[list[int]],
            slot_ids: list[int],
            graph: DecodeGraph
    ) -> list[int]:

        if not slot_ids:
            return []

        kv_cache = graph.kv_cache
        batch_size, unused_slot = kv_cache.batch_size, kv_cache.unused_slot

        if any(kv_cache.occupancy[slot_ids].tolist()):
            return []

        if len(prompts) != len(slot_ids):
            raise ValueError(
                f"len(prompts) must equal len(slot_ids)."
            )

        padded_slot_ids = pad_slot_ids(slot_ids, batch_size, unused_slot)
        kv_cache.reset(slot_ids)

        prompts_size = len(prompts)
        prompts = prompts + [[0] for _ in range(batch_size - prompts_size)]
        lens_list = list(map(len, prompts))

        kv_cache.prepare_prefill(lens_list, padded_slot_ids)

        input_ids = torch.tensor([
            id_
            for p in prompts
            for id_ in p
        ], dtype=torch.long, device=kv_cache.device).unsqueeze(0)

        position_ids = kv_cache.pos[None]

        total_T = sum(lens_list)

        with torch.inference_mode():
            if total_T > 1:
                torch._dynamo.mark_dynamic(input_ids, 1)
                torch._dynamo.mark_dynamic(position_ids, 1)
                torch._dynamo.mark_dynamic(kv_cache.active_slot_ids, 0)
                torch._dynamo.mark_dynamic(kv_cache.pos, 0)

            logits = self._prefill_forward(
                input_ids=input_ids,
                position_ids=position_ids,
                kv_cache=kv_cache
            )[:prompts_size]

            next_tokens = torch.argmax(
                logits,
                dim=-1,
            )

            kv_cache.last_generated[slot_ids] = next_tokens
            kv_cache.advance()

        return next_tokens.tolist()






















