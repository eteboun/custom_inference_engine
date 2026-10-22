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
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.slot_ids.device)
        s = len(ids)
        self.slot_ids[:s] = ids
        self.slot_ids[s:] = 0

    def write_lens(self, lens: list[int] | torch.Tensor):
        lens = torch.as_tensor(lens, dtype=torch.long, device=self.lens.device)
        s = len(lens)
        self.lens[:s] = lens
        self.lens[s:] = 0

    def write_cu_seqlens(self, cu_seqlens: list[int] | torch.Tensor):
        cu_seqlens = torch.as_tensor(cu_seqlens, dtype=torch.long, device=self.cu_seqlens.device)
        s = len(cu_seqlens)
        self.cu_seqlens[:s] = cu_seqlens
        self.cu_seqlens[s:] = cu_seqlens[-1]

