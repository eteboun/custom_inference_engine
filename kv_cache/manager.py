import torch

class KVCacheManager:

    def __init__(self, config, max_cache_len, device, dtype):

        self.config = config
        self.max_cache_len = max_cache_len
        self.device = device
        self.dtype = dtype

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
                size=(self.num_kv_heads, max_cache_len, self.head_dim),
                device=device,
                dtype=dtype,
            ) for _ in range(self.num_layers)
        ]

        self.values = [
            torch.zeros(
                size=(self.num_kv_heads, max_cache_len, self.head_dim),
                device=device,
                dtype=dtype,
            ) for _ in range(self.num_layers)
        ]

        self.start = torch.tensor(
            [0], device=device, dtype=torch.long
        )

    def update(
            self,
            layer_num: int,
            k_proj: torch.Tensor,
            v_proj: torch.Tensor
    ) -> None:
        L = k_proj.size(0)

        kv_indices = self.start + torch.arange(L, device=self.device)

        k_cache = self.keys[layer_num]
        k_new = k_proj.reshape(
            L, self.num_kv_heads, self.head_dim
        ).transpose(0, 1)

        v_cache = self.values[layer_num]
        v_new = v_proj.reshape(
            L, self.num_kv_heads, self.head_dim
        ).transpose(0, 1)

        k_cache.index_copy_(1, kv_indices, k_new)
        v_cache.index_copy_(1, kv_indices, v_new)

    def advance(self, L: int) -> None:
        self.start += L

    def reset(self) -> None:
        self.start.zero_()


