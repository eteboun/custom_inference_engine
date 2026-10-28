import torch

class AttentionMetadata:

    def __init__(
            self,
            batch_size: int,
            max_seqlen: int,
            full_decode: bool,
            device: torch.device
    ):
        self.slot_ids = torch.zeros(
            (batch_size,), dtype=torch.long, device=device
        )
        self.cu_seqlens = torch.zeros(
            (batch_size + 1,), dtype=torch.long, device=device
        )

        self.pages = None
        self.offsets = None

        self.max_seqlen = max_seqlen
        self.full_decode = full_decode

    def write_slot_ids(self, ids: list[int] | torch.Tensor):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.slot_ids.device)
        s = len(ids)
        self.slot_ids[:s] = ids
        self.slot_ids[s:] = 0

    def write_cu_seqlens(self, cu_seqlens: list[int] | torch.Tensor):
        cu_seqlens = torch.as_tensor(cu_seqlens, dtype=torch.long, device=self.cu_seqlens.device)
        s = len(cu_seqlens)
        self.cu_seqlens[:s] = cu_seqlens
        self.cu_seqlens[s:] = cu_seqlens[-1]

