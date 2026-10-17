import torch

class KVCacheManager:

    def __init__(
            self, config,
            max_cache_len, batch_size,
            device, dtype
    ):

        self.config = config
        self.max_cache_len = max_cache_len
        self.batch_size = batch_size
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
                size=(batch_size, self.num_kv_heads,
                      max_cache_len, self.head_dim),
                device=device,
                dtype=dtype,
            ) for _ in range(self.num_layers)
        ]

        self.values = [
            torch.zeros(
                size=(batch_size, self.num_kv_heads,
                      max_cache_len, self.head_dim),
                device=device,
                dtype=dtype,
            ) for _ in range(self.num_layers)
        ]

        self.start = torch.zeros(
            size=(batch_size,), device=device, dtype=torch.long
        )

        self._batch_range = torch.arange(
            self.batch_size,
            device=self.device
        )[:, None, None]

        self._head_range = torch.arange(
            self.num_kv_heads,
            device=self.device
        )[None, :, None]

    def update(
            self,
            layer_num: int,
            k_proj: torch.Tensor,
            v_proj: torch.Tensor
    ) -> None:
        L = k_proj.size(1)

        k_cache = self.keys[layer_num]
        k_new = k_proj.reshape(
            self.batch_size, L, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        v_cache = self.values[layer_num]
        v_new = v_proj.reshape(
            self.batch_size, L, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        pos = (
                self.start[:, None]
                + torch.arange(L, device=self.device)[None, :]
            )[:, None, :]

        k_cache.index_put_(
            (self._batch_range, self._head_range, pos),
            k_new
        )

        v_cache.index_put_(
            (self._batch_range, self._head_range, pos),
            v_new
        )

    def advance(self, lengths: torch.Tensor) -> None:
        self.start += lengths

    def reset(self) -> None:
        self.start.zero_()


