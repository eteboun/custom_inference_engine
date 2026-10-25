import torch

from attn.core import use_triton_attention
from attn.metadata import AttentionMetadata
from scheduler.slots import pad_slot_ids
from kv_cache.storage import KVStorage

class DecodeModule:

    def __init__(self, model, kv_cache: KVStorage, batch_size: int, warmup_iters: int = 3):
        self.model = model
        self.batch_size = batch_size
        self.kv_cache = kv_cache

        device = model.device
        self.device = device

        self.metadata = AttentionMetadata(
            batch_size=batch_size,
            max_seqlen=batch_size,
            device=device,
            is_decode=True
        )

        with torch.inference_mode():
            self.static_token = torch.zeros(
                (1, batch_size), device=device, dtype=torch.long
            )
            self.static_position_ids = torch.zeros(
                (1, batch_size), device=device, dtype=torch.long
            )

            self.decode_lens = torch.ones(
                (batch_size,), dtype=torch.long, device=device
            )

            use_triton_attention(model)

            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                self.prepare_decode()
                self._forward()

                forward = torch.compile(
                    self._forward, fullgraph=True, dynamic=False
                )

                for _ in range(warmup_iters):
                    self.prepare_decode()
                    forward()
            torch.cuda.current_stream().wait_stream(side_stream)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.prepare_decode()
                self.static_logits = forward().logits[0]

        self.warmup()

    def warmup(self) -> None:
        slot_ids = list(range(1, min(self.batch_size, self.kv_cache.n_slots - 1) + 1))
        self.forward(slot_ids)
        self.kv_cache.reset(slot_ids)

    def _forward(self):
        outputs = self.model(
            input_ids=self.static_token,
            position_ids=self.static_position_ids,
            kv_manager=self.kv_cache,
            logits_to_keep=0,
            use_cache=False,
            metadata=self.metadata
        )
        self.kv_cache.advance(
            slot_ids=self.metadata.slot_ids,
            lens=self.metadata.lens
        )
        return outputs

    def upload(
            self,
            slot_ids: list[int]
    ) -> None:
        padded_slot_ids = pad_slot_ids(
            slot_ids, self.batch_size, self.kv_cache.unused_idx
        )
        self.static_position_ids.copy_(
            self.kv_cache.start[padded_slot_ids][None]
        )
        self.static_token.copy_(
            self.kv_cache.last_generated[padded_slot_ids][None]
        )
        self.metadata.write_slot_ids(slot_ids)

    def prepare_decode(self):
        self.metadata.write_lens(self.decode_lens)
        self.metadata.pos = self.kv_cache.start[self.metadata.slot_ids]
        self.metadata.active_slot_ids = self.metadata.slot_ids

    def replay(self):
        self.graph.replay()
        return self.static_logits

    def forward(
            self,
            slot_ids: list[int]
    ) -> list[int]:

        if not slot_ids:
            return []

        kv_cache = self.kv_cache

        with torch.inference_mode():
            self.upload(slot_ids)
            logits = self.replay()
            next_tokens = torch.argmax(logits, dim=-1)[:len(slot_ids)]

            kv_cache.last_generated[slot_ids] = next_tokens

        return next_tokens.tolist()

