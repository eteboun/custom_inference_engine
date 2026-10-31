import torch

from kv_cache.storage import KVStorage
from scheduler.decode_module import DecodeModule
from scheduler.mixed_module import FullMixedModule, PiecewiseMixedModule


def _smallest_fit(sizes, value: int) -> int:
    for size in sizes:
        if size >= value:
            return size

    raise RuntimeError(
        "Invalid size."
    )


class FullGraphBackend:

    def __init__(
            self,
            model,
            kv_cache: KVStorage,
            batch_sizes: list[int],
            token_counts: list[int]
    ) -> None:
        self.decode_modules = {
            batch_size: DecodeModule(
                model=model,
                kv_cache=kv_cache,
                batch_size=batch_size
            )
            for batch_size in batch_sizes
        }

        self.mixed_modules = {
            batch_size: {
                token_count: FullMixedModule(
                    model=model,
                    kv_cache=kv_cache,
                    batch_size=batch_size,
                    token_count=token_count
                )
                for token_count in token_counts
                if token_count >= batch_size or token_count == token_counts[-1]
            }
            for batch_size in batch_sizes
        }

    def get_decode_module(self, batch_size: int) -> DecodeModule:
        return self.decode_modules[
            _smallest_fit(self.decode_modules, batch_size)
        ]

    def get_mixed_module(self, batch_size: int, token_count: int) -> FullMixedModule:
        row = self.mixed_modules[
            _smallest_fit(self.mixed_modules, batch_size)
        ]
        return row[_smallest_fit(row, token_count)]


class PiecewiseGraphBackend:

    def __init__(
            self,
            model,
            kv_cache: KVStorage,
            token_counts: list[int]
    ) -> None:
        self.buffer = torch.zeros(
            (1, max(token_counts),
             kv_cache.num_q_heads, kv_cache.head_dim),
            dtype=model.dtype,
            device=model.device
        )

        self.mixed_modules = {
            token_count: PiecewiseMixedModule(
                model=model,
                kv_cache=kv_cache,
                token_count=token_count,
                buffer=self.buffer
            )
            for token_count in token_counts
        }

    def get_mixed_module(self, token_count: int) -> PiecewiseMixedModule:
        return self.mixed_modules[
            _smallest_fit(self.mixed_modules, token_count)
        ]
