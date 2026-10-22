import torch

class KVStorage:

    def __init__(
            self, config,
            max_cache_len, n_slots,
            device, dtype
    ):
        self.config = config
        self.max_cache_len = max_cache_len
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

        self.last_generated = torch.zeros(
            size=(n_slots,), dtype=torch.long, device=self.device
        )

        self.occupancy = torch.zeros(
            size=(n_slots,), dtype=torch.bool, device=self.device
        )

    def update(
            self,
            layer_num: int,
            active_slot_ids: torch.Tensor,
            pos: torch.Tensor,
            k_proj: torch.Tensor,
            v_proj: torch.Tensor
    ) -> None:
        k_cache = self.keys[layer_num]
        k_cache[active_slot_ids, :, pos] = k_proj.view(
            -1, self.num_kv_heads, self.head_dim
        )

        v_cache = self.values[layer_num]
        v_cache[active_slot_ids, :, pos] = v_proj.view(
            -1, self.num_kv_heads, self.head_dim
        )

    def advance(
            self,
            slot_ids: torch.Tensor,
            lens: torch.Tensor
    ) -> None:
        self.start[slot_ids] += lens
        self.occupancy[slot_ids] = True
        self.start[self.unused_slot] = 0
        self.occupancy[self.unused_slot] = False

    def validate_reset(self, slot_ids: list[int]):
        if any(not 0 < slot < self.n_slots for slot in slot_ids):
            raise ValueError(f"slot_ids'te [1, {self.n_slots}) dışında slot var")

    def reset(self, slot_ids: list[int]) -> None:
        self.validate_reset(slot_ids)
        self.occupancy[slot_ids] = False
        self.start[slot_ids] = 0


