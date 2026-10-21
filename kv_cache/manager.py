import torch
from torch.nn import functional as F

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

        self.decode_lens = torch.ones(
            self.batch_size, dtype=torch.long, device=self.device
        )

        self.decode_seq_idx = torch.arange(
            self.batch_size, dtype=torch.long, device=self.device
        )

        self.pos = None
        self.seq_idx = None
        self.cu_seqlens = None
        self.lens = None
        self.max_seqlen = 0
        self.is_decode = False

    def prepare_prefill(
            self,
            lens_list: list[int]
    ):
        total_T = sum(lens_list)
        max_seqlen = max(lens_list)
        lens = torch.tensor(lens_list, dtype=torch.long, device=self.device)

        cu_seqlens = F.pad(lens.cumsum(0), (1, 0))
        seq_idx = torch.repeat_interleave(
            torch.arange(self.batch_size, device=self.device),
            lens,
            output_size=total_T
        )

        offsets = torch.arange(total_T, device=self.device) - cu_seqlens[seq_idx]
        pos = self.start[seq_idx] + offsets

        self.pos = pos
        self.seq_idx = seq_idx
        self.is_decode = False
        self.cu_seqlens = cu_seqlens
        self.lens = lens
        self.max_seqlen = max_seqlen

    def prepare_decode(self):
        self.lens = self.decode_lens
        self.seq_idx = self.decode_seq_idx
        self.pos = self.start
        self.is_decode = True

    def update(
            self,
            layer_num: int,
            k_proj: torch.Tensor,
            v_proj: torch.Tensor
    ) -> None:
        k_cache = self.keys[layer_num]
        k_cache[self.seq_idx, :,self.pos] = k_proj.view(
            -1, self.num_kv_heads, self.head_dim
        )

        v_cache = self.values[layer_num]
        v_cache[self.seq_idx, :, self.pos] = v_proj.view(
            -1, self.num_kv_heads, self.head_dim
        )

    def advance(self) -> None:
        self.start += self.lens

    def reset(self) -> None:
        self.start.zero_()


