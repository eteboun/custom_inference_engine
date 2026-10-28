import torch
import torch.nn.functional as F

from attn.metadata import AttentionMetadata
from kv_cache.storage import KVStorage
from scheduler.slots import pad_slot_ids

class MixedModule:

    def __init__(
            self,
            model,
            kv_cache: KVStorage,
            batch_size: int,
            token_count: int,
            warmup_iters: int = 3
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.token_count = token_count
        self.kv_cache = kv_cache

        device = model.device
        self.device = device

        self.metadata = AttentionMetadata(
            batch_size=batch_size,
            token_count=token_count,
            device=device,
            full_decode=False
        )

        with torch.inference_mode():
            self.lens = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            self.static_input_ids = torch.zeros(
                (1, token_count), device=device, dtype=torch.long
            )
            self.static_position_ids = torch.zeros(
                (1, token_count), device=device, dtype=torch.long
            )

            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                self._forward()

                forward = torch.compile(
                    self._forward, fullgraph=True, dynamic=False
                )

                for _ in range(warmup_iters):
                    forward()

            torch.cuda.current_stream().wait_stream(side_stream)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_logits = forward()

        self.warmup()

    def warmup(self) -> None:
        n = min(self.batch_size, self.kv_cache.n_slots - 1, self.token_count)
        slot_ids = list(range(1, n + 1))

        prompts = [[0] * (self.token_count // n) for _ in slot_ids]
        self.forward(prompts, slot_ids)

        self.kv_cache.reset(slot_ids)

    def _forward(self) -> torch.Tensor:
        hidden = self.model.model(
            input_ids=self.static_input_ids,
            position_ids=self.static_position_ids,
            kv_manager=self.kv_cache,
            use_cache=False,
            metadata=self.metadata
        ).last_hidden_state
        self.kv_cache.mixed_advance(
            slot_ids=self.metadata.slot_ids,
            lens=self.lens
        )
        return self.model.lm_head(hidden[0, self.metadata.cu_seqlens[1:] - 1])

    def upload(
            self,
            prompts: list[list[int]],
            slot_ids: list[int]
    ) -> None:

        padded_slot_ids = pad_slot_ids(
            slot_ids, self.batch_size, self.kv_cache.unused_idx
        )

        padded_prompts = prompts + [[] for _ in range(self.batch_size - len(prompts))]

        lens_list = list(
            map(lambda x: len(x), padded_prompts)
        )

        total_T = sum(lens_list)
        lens = torch.tensor(
            lens_list, dtype=torch.long, device=self.device
        )
        slot_ids = torch.tensor(
            padded_slot_ids, dtype=torch.long, device=self.device
        )

        cu_seqlens = F.pad(lens.cumsum(0), (1, 0))
        extended_slot_ids = torch.repeat_interleave(
            slot_ids,
            lens,
            dim=-1,
            output_size=total_T
        )

        seq_offsets = torch.arange(total_T, device=self.device) - torch.repeat_interleave(
            cu_seqlens[:-1],
            lens,
            dim=-1,
            output_size=total_T
        )

        input_ids = torch.tensor(
            [
                id_
                for p in prompts
                for id_ in p
            ], dtype=torch.long, device=self.kv_cache.device
        )

        pos = self.kv_cache.lens[extended_slot_ids] + seq_offsets

        self.static_input_ids[0, :total_T] = input_ids
        self.static_input_ids[0, total_T:] = 0
        self.static_position_ids[0, :total_T] = pos
        self.static_position_ids[0, total_T:] = 0
        self.lens.copy_(lens)

        self.metadata.write_pages(self.kv_cache.block_table[
            extended_slot_ids, pos // self.kv_cache.page_size
        ])
        self.metadata.write_offsets(pos % self.kv_cache.page_size)
        self.metadata.write_cu_seqlens(cu_seqlens)
        self.metadata.write_slot_ids(slot_ids)

    def replay(self):
        self.graph.replay()
        return self.static_logits

    def forward(
            self,
            prompts: list[list[int]],
            slot_ids: list[int]
    ) -> list[int]:

        if len(prompts) != len(slot_ids):
            raise ValueError(
                "len(prompts) must equal len(slot_ids)."
            )

        if not slot_ids:
            return []

        with torch.inference_mode():
            self.upload(
                prompts=prompts,
                slot_ids=slot_ids
            )
            logits = self.replay()
            next_tokens = torch.argmax(logits, dim=-1)[:len(slot_ids)]

            self.kv_cache.last_generated[slot_ids] = next_tokens

        return next_tokens.tolist()