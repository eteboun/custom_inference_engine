import math
import torch

class KVStorage:

    def __init__(
            self, config,
            max_cache_len,
            page_size, n_slots,
            device, dtype
    ):
        self.config = config
        self.max_cache_len = max_cache_len
        self.page_size = page_size
        self.n_slots = n_slots
        self.device = device
        self.dtype = dtype

        self.n_pages = math.ceil(max_cache_len / page_size)
        self.unused_idx = 0

        self.num_layers = config.num_hidden_layers
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads
        )

        self.keys = [
            torch.zeros(
                size=(self.n_pages, self.num_kv_heads,
                      self.page_size, self.head_dim),
                device=device,
                dtype=dtype,
            ) for _ in range(self.num_layers)
        ]

        self.values = [
            torch.zeros(
                size=(self.n_pages, self.num_kv_heads,
                      self.page_size, self.head_dim),
                device=device,
                dtype=dtype,
            ) for _ in range(self.num_layers)
        ]

        self.lens = torch.zeros(
            size=(n_slots,), dtype=torch.long, device=self.device
        )

        self.block_table = torch.zeros(
            size=(n_slots, self.n_pages), dtype=torch.long, device=self.device
        )

        self.last_generated = torch.zeros(
            size=(n_slots,), dtype=torch.long, device=self.device
        )

    def prefill_update(
            self,
            layer_num: int,
            pages: torch.Tensor,
            offsets: torch.Tensor,
            k_proj: torch.Tensor,
            v_proj: torch.Tensor
    ) -> None:

        # (n_pages, H_kv, page_size, head_dim)
        k_cache = self.keys[layer_num]

        # (totalT, H_kv, head_dim)
        k_cache[pages, :, offsets] = k_proj.view(
            -1, self.num_kv_heads, self.head_dim
        )

        # (n_pages, H_kv, page_size, head_dim)
        v_cache = self.values[layer_num]

        # (totalT, H_kv, head_dim)
        v_cache[pages, :, offsets] = v_proj.view(
            -1, self.num_kv_heads, self.head_dim
        )

    def decode_update(
            self,
            layer_num: int,
            slot_ids: torch.Tensor,
            k_proj: torch.Tensor,
            v_proj: torch.Tensor
    ) -> None:

        lens = self.lens[slot_ids]

        pages = self.block_table[slot_ids, lens // self.page_size]
        starts = lens % self.page_size

        # (n_pages, H_kv, page_size, head_dim)
        k_cache = self.keys[layer_num]

        # (batch_size, H_kv, head_dim)
        k_cache[
            pages,
            :,
            starts
        ] = k_proj.view(
            -1, self.num_kv_heads, self.head_dim
        )

        # (n_pages, H_kv, page_size, head_dim)
        v_cache = self.values[layer_num]

        # (batch_size, H_kv, head_dim)
        v_cache[
            pages,
            :,
            starts
        ] = v_proj.view(
            -1, self.num_kv_heads, self.head_dim
        )

    def decode_advance(
            self,
            slot_ids: torch.Tensor
    ) -> None:
        self.lens[slot_ids] += 1
        self.lens[self.unused_idx] = 0

    def prefill_advance(
            self,
            slot_ids: torch.Tensor,
            lens: torch.Tensor
    ) -> None:
        self.lens[slot_ids] += lens
        self.lens[self.unused_idx] = 0

    def reset(self, slot_ids: list[int]) -> None:
        self.lens[slot_ids] = 0
        self.block_table[slot_ids] = 0
