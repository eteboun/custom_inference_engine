import torch

class AttentionMetadata:

    def __init__(
            self,
            batch_size: int,
            max_seqlen: int,
            is_decode: bool,
            device: torch.device
    ):
        self.slot_ids = torch.zeros(
            (batch_size,), dtype=torch.long, device=device
        )
        self.lens = torch.zeros(
            (batch_size,), dtype=torch.long, device=device
        )
        self.cu_seqlens = torch.zeros(
            (batch_size + 1,), dtype=torch.long, device=device
        )

        self.pos = None
        self.active_slot_ids = None
        self.max_seqlen = max_seqlen

        self.is_decode = is_decode

    def write_slot_ids(self, ids: list[int] | torch.Tensor):
        s = len(ids)
        self.slot_ids[:s] = ids
        self.slot_ids[s:] = 0

