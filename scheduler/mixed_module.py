import torch
import torch.nn.functional as F

import random

from attn.metadata import AttentionMetadata
from kv_cache.storage import KVStorage
from scheduler.slots import pad_slot_ids

class MixedModule:

    def __init__(self, model, kv_cache: KVStorage, batch_size: int, warmup_iters: int = 3):
        self.model = model
        self.batch_size = batch_size
        self.kv_cache = kv_cache

        device = model.device
        self.device = device

        self.metadata = AttentionMetadata(
            batch_size=batch_size,
            max_seqlen=0,
            device=device,
            full_decode=False
        )

        if batch_size > 1:
            torch._dynamo.mark_dynamic(self.metadata.cu_seqlens, 0)
            torch._dynamo.mark_dynamic(self.metadata.slot_ids, 0)

        self.warmup(warmup_iters)

    def warmup(self, warmup_iters: int) -> None:
        slot_ids = list(range(1, min(self.batch_size, self.kv_cache.n_slots - 1) + 1))
        max_len = min(128, self.kv_cache.max_cache_len)
        rng = random.Random(0)

        for _ in range(warmup_iters):
            prompts = [[0] * rng.randint(2, max_len) for _ in slot_ids]
            self.forward(prompts, slot_ids)

            self.kv_cache.reset(slot_ids)

    @torch.compile(fullgraph=True)
    def _mixed_forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache: KVStorage
    ) -> torch.Tensor:
        hidden = self.model.model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_manager=kv_cache,
            use_cache=False,
            metadata=self.metadata
        ).last_hidden_state
        return self.model.lm_head(hidden[0, self.metadata.cu_seqlens[1:] - 1])

    def validate_mixed(
            self,
            lens_list: list[int],
            slot_ids: list[int]
    ):
        if len(lens_list) != self.batch_size:
            raise ValueError(
                f"lens_list uzunluğu ({len(lens_list)}) batch_size'a ({self.batch_size}) eşit değil"
            )
        if len(slot_ids) != self.batch_size:
            raise ValueError(
                f"slot_ids uzunluğu ({len(slot_ids)}) batch_size'a ({self.batch_size}) eşit değil"
            )
        if any(length <= 0 for length in lens_list):
            raise ValueError("lens_list'te pozitif olmayan uzunluk var")
        if any(length > self.kv_cache.max_cache_len for length in lens_list):
            raise ValueError(
                f"lens_list'te max_cache_len'i ({self.kv_cache.max_cache_len}) aşan uzunluk var"
            )
        if any(not 0 <= slot < self.kv_cache.n_slots for slot in slot_ids):
            raise ValueError(f"slot_ids'te [0, {self.kv_cache.n_slots}) dışında slot var")

        real_slots = [slot for slot in slot_ids if slot != 0]
        if len(real_slots) != len(set(real_slots)):
            raise ValueError("slot_ids'te tekrar eden slot var")

    def prepare_mixed(
            self,
            lens_list: list[int],
            slot_ids: list[int]
    ):
        self.validate_mixed(lens_list, slot_ids)

        total_T = sum(lens_list)
        max_seqlen = max(lens_list)
        lens = torch.tensor(
            lens_list, dtype=torch.long, device=self.device
        )
        slot_ids = torch.tensor(
            slot_ids, dtype=torch.long, device=self.device
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

        pos = self.kv_cache.lens[extended_slot_ids] + seq_offsets

        self.metadata.pages = self.kv_cache.block_table[
            extended_slot_ids, pos // self.kv_cache.page_size
        ]
        self.metadata.offsets = pos % self.kv_cache.page_size
        self.metadata.max_seqlen = max_seqlen

        self.metadata.write_cu_seqlens(cu_seqlens)
        self.lens = lens
        self.metadata.write_slot_ids(slot_ids)

        return pos

    def forward(
            self,
            prompts: list[list[int]],
            slot_ids: list[int]
    ) -> list[int]:

        if not slot_ids:
            return []

        unused_slot = self.kv_cache.unused_idx

        if len(prompts) != len(slot_ids):
            raise ValueError(
                "len(prompts) must equal len(slot_ids)."
            )

        padded_slot_ids = pad_slot_ids(slot_ids, self.batch_size, unused_slot)

        prompts_size = len(prompts)
        prompts = prompts + [[0] for _ in range(self.batch_size - prompts_size)]
        lens_list = list(map(len, prompts))

        input_ids = torch.tensor([
            id_
            for p in prompts
            for id_ in p
        ], dtype=torch.long, device=self.kv_cache.device).unsqueeze(0)

        position_ids = self.prepare_mixed(lens_list, padded_slot_ids)[None]

        with torch.inference_mode():
            if self.batch_size > 1:
                torch._dynamo.mark_dynamic(input_ids, 1)
                torch._dynamo.mark_dynamic(position_ids, 1)
                torch._dynamo.mark_dynamic(self.metadata.pages, 0)
                torch._dynamo.mark_dynamic(self.metadata.offsets, 0)

            logits = self._mixed_forward(
                input_ids=input_ids,
                position_ids=position_ids,
                kv_cache=self.kv_cache
            )[:prompts_size]

            next_tokens = torch.argmax(
                logits,
                dim=-1,
            )

            self.kv_cache.last_generated[slot_ids] = next_tokens
            self.kv_cache.mixed_advance(
                slot_ids=self.metadata.slot_ids,
                lens=self.lens
            )

        return next_tokens.tolist()
