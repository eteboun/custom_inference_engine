from typing import ClassVar
from scheduler.decode_module import use_triton_attention

class Scheduler:

    DECODE_BATCH_BUCKETS: ClassVar[tuple] = (1, 2, 4, 8, 16)
    KV_CACHE_CAPACITY: ClassVar[int] = 32768

    def __init__(
            self,
            model,
            eos_token_ids: set[int],
            max_new_tokens: int,
            n_slots: int
    ) -> None:
        if max_new_tokens > self.KV_CACHE_CAPACITY:
            raise ValueError(
                f"max_new_tokens can't be larger than {self.KV_CACHE_CAPACITY}."
            )

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive."
            )

        if not eos_token_ids:
            raise ValueError(
                "eos_token_ids must not be empty."
            )

        if n_slots <= 0:
            raise ValueError(
                "n_slots must be positive."
            )

        try:
            use_triton_attention(model)

        except Exception as exc:
            raise RuntimeError(
                "Failed to switch triton attention:"
                f" {exc}"
            )

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.n_slots = n_slots
        self.eos_token_ids = frozenset(eos_token_ids)















