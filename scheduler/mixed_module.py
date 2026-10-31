import torch
import torch.nn.functional as F

from attn.metadata import AttentionMetadata
from kv_cache.storage import KVStorage
from piecewise.backend import get_piecewise_backend
from scheduler.slots import pad_slot_ids

class MixedModule:

    def __init__(
            self,
            model,
            kv_cache: KVStorage,
            token_count: int
    ) -> None:
        self.model = model
        self.token_count = token_count
        self.kv_cache = kv_cache

        device = model.device
        self.device = device

        self.metadata = AttentionMetadata(
            token_count=token_count,
            device=device,
            full_decode=False
        )

        self.lens = None

        with torch.inference_mode():
            self.static_input_ids = torch.zeros(
                (1, token_count), device=device, dtype=torch.long
            )
            self.static_position_ids = torch.zeros(
                (1, token_count), device=device, dtype=torch.long
            )

    def warmup(self, n: int) -> None:
        slot_ids = list(range(1, n + 1))

        prompts = [[0] * (self.token_count // n) for _ in slot_ids]
        self.forward(prompts, slot_ids)

        self.kv_cache.reset(slot_ids)

    def _forward(self) -> torch.Tensor:
        return self.model.model(
            input_ids=self.static_input_ids,
            position_ids=self.static_position_ids,
            kv_manager=self.kv_cache,
            use_cache=False,
            metadata=self.metadata
        ).last_hidden_state

    def _result(self, hidden: torch.Tensor) -> torch.Tensor:
        self.kv_cache.mixed_advance(
            slot_ids=self.metadata.slot_ids,
            lens=self.lens
        )
        return self.model.lm_head(
            hidden[0, self.metadata.cu_seqlens[1:] - 1]
        )

    def _pad(self, slot_ids: list[int]) -> list[int]:
        raise NotImplementedError

    def _write_batch(self, slot_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
        raise NotImplementedError

    def replay(self) -> torch.Tensor:
        raise NotImplementedError

    def upload(
            self,
            prompts: list[list[int]],
            slot_ids: list[int]
    ) -> None:

        padded_slot_ids = self._pad(slot_ids)

        padded_prompts = prompts + [[] for _ in range(len(padded_slot_ids) - len(prompts))]

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
        self.lens = lens

        self.metadata.write_pages(self.kv_cache.block_table[
            extended_slot_ids, pos // self.kv_cache.page_size
        ])
        self.metadata.write_offsets(pos % self.kv_cache.page_size)
        self._write_batch(slot_ids, cu_seqlens)

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
            hidden = self.replay()
            logits = self._result(hidden)

            next_tokens = torch.argmax(logits, dim=-1)[:len(slot_ids)]

            self.kv_cache.last_generated[slot_ids] = next_tokens

        return next_tokens.tolist()


class FullMixedModule(MixedModule):

    def __init__(
            self,
            model,
            kv_cache: KVStorage,
            batch_size: int,
            token_count: int,
            warmup_iters: int = 3
    ) -> None:
        super().__init__(model, kv_cache, token_count)
        self.batch_size = batch_size
        self.metadata.prepare_full_graph(batch_size)

        with torch.inference_mode():
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
                self.static_hidden = forward()

        self.warmup(min(batch_size, kv_cache.n_slots - 1, token_count))

    def _pad(self, slot_ids: list[int]) -> list[int]:
        return pad_slot_ids(
            slot_ids, self.batch_size, self.kv_cache.unused_idx
        )

    def _write_batch(self, slot_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
        self.metadata.write_cu_seqlens(cu_seqlens)
        self.metadata.write_slot_ids(slot_ids)

    def replay(self) -> torch.Tensor:
        self.graph.replay()
        return self.static_hidden


class PiecewiseMixedModule(MixedModule):

    def __init__(
            self,
            model,
            kv_cache: KVStorage,
            token_count: int,
            buffer: torch.Tensor
    ) -> None:
        super().__init__(model, kv_cache, token_count)

        self.compiled = torch.compile(
            self._forward,
            backend=get_piecewise_backend(buffer[:, :token_count]),
            fullgraph=True,
            dynamic=False
        )

        self.warmup(min(kv_cache.n_slots - 1, token_count))

    def _pad(self, slot_ids: list[int]) -> list[int]:
        return slot_ids

    def _write_batch(self, slot_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
        if len(slot_ids) > 1:
            torch._dynamo.mark_dynamic(slot_ids, 0)
            torch._dynamo.mark_dynamic(cu_seqlens, 0)
        self.metadata.slot_ids = slot_ids
        self.metadata.cu_seqlens = cu_seqlens

    def replay(self) -> torch.Tensor:
        return self.compiled()
