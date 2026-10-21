import torch
from torch.nn import functional as F

class KVCacheManager:

    def __init__(
            self, config,
            max_cache_len, batch_size, n_slots,
            device, dtype
    ):
        self.config = config
        self.max_cache_len = max_cache_len
        self.batch_size = batch_size
        self.n_slots = n_slots
        self.device = device
        self.dtype = dtype

        self.unused_slot = 0

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
                size=(n_slots, self.num_kv_heads,
                      max_cache_len, self.head_dim),
                device=device,
                dtype=dtype,
            ) for _ in range(self.num_layers)
        ]

        self.values = [
            torch.zeros(
                size=(n_slots, self.num_kv_heads,
                      max_cache_len, self.head_dim),
                device=device,
                dtype=dtype,
            ) for _ in range(self.num_layers)
        ]

        self.start = torch.zeros(
            size=(n_slots,), device=device, dtype=torch.long
        )

        self.decode_lens = torch.ones(
            (batch_size,), dtype=torch.long, device=self.device
        )

        self.slot_ids = torch.zeros(
            (batch_size,), dtype=torch.long, device=self.device
        )

        self.last_generated = torch.zeros(
            size=(n_slots,), dtype=torch.long, device=self.device
        )

        self.pos = None
        self.cu_seqlens = None
        self.active_slot_ids = None
        self.lens = None
        self.max_seqlen = 0
        self.is_decode = False

        self.occupancy = torch.zeros(
            size=(n_slots,), dtype=torch.bool, device=self.device
        )

    def _write_slot_ids_tensor(self, ids: torch.Tensor):
        s = ids.shape[-1]
        self.slot_ids[:s] = ids
        self.slot_ids[s:] = 0

    def write_slot_ids(self, ids: list[int]):
        s = len(ids)
        self.slot_ids[:s] = ids
        self.slot_ids[s:] = 0

    def validate_prefill(
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
        if any(length > self.max_cache_len for length in lens_list):
            raise ValueError(
                f"lens_list'te max_cache_len'i ({self.max_cache_len}) aşan uzunluk var"
            )
        if any(not 0 <= slot < self.n_slots for slot in slot_ids):
            raise ValueError(f"slot_ids'te [0, {self.n_slots}) dışında slot var")

        real_slots = [slot for slot in slot_ids if slot != 0]
        if len(real_slots) != len(set(real_slots)):
            raise ValueError("slot_ids'te tekrar eden slot var")

    def prepare_prefill(
            self,
            lens_list: list[int],
            slot_ids: list[int]
    ):
        self.validate_prefill(lens_list, slot_ids)

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

        offsets = torch.arange(total_T, device=self.device) - torch.repeat_interleave(
            cu_seqlens[:-1],
            lens,
            dim=-1,
            output_size=total_T
        )

        pos = self.start[extended_slot_ids] + offsets

        self.pos = pos
        self.cu_seqlens = cu_seqlens
        self.lens = lens

        self._write_slot_ids_tensor(slot_ids)

        self.active_slot_ids = extended_slot_ids
        self.max_seqlen = max_seqlen
        self.is_decode = False

    def prepare_decode(self):
        self.lens = self.decode_lens
        self.pos = self.start[self.slot_ids]
        self.active_slot_ids = self.slot_ids
        self.is_decode = True

    def update(
            self,
            layer_num: int,
            k_proj: torch.Tensor,
            v_proj: torch.Tensor
    ) -> None:
        k_cache = self.keys[layer_num]
        k_cache[self.active_slot_ids, :, self.pos] = k_proj.view(
            -1, self.num_kv_heads, self.head_dim
        )

        v_cache = self.values[layer_num]
        v_cache[self.active_slot_ids, :, self.pos] = v_proj.view(
            -1, self.num_kv_heads, self.head_dim
        )

    def advance(self) -> None:
        self.start[self.slot_ids] += self.lens
        self.occupancy[self.slot_ids] = True
        self.start[self.unused_slot] = 0
        self.occupancy[self.unused_slot] = False

    def validate_reset(self, slot_ids: list[int]):
        if any(not 0 < slot < self.n_slots for slot in slot_ids):
            raise ValueError(f"slot_ids'te [1, {self.n_slots}) dışında slot var")

    def reset(self, slot_ids: list[int]) -> None:
        self.validate_reset(slot_ids)
        self.occupancy[slot_ids] = False
        self.start[slot_ids] = 0


