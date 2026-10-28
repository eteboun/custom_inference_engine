from kv_cache.storage import KVStorage
from scheduler.decode_module import DecodeModule
from scheduler.mixed_module import MixedModule

class Bucket:
    def __init__(self, model, kv_cache: KVStorage, batch_size: int, token_counts: list[int]):
        self.batch_size = batch_size

        self.decode_module = DecodeModule(
            model=model,
            kv_cache=kv_cache,
            batch_size=batch_size
        )

        self.mixed_modules = {
            token_count: MixedModule(
                model=model,
                kv_cache=kv_cache,
                batch_size=batch_size,
                token_count=token_count
            )
            for token_count in token_counts
            if token_count >= batch_size or token_count == token_counts[-1]
        }

    def get_mixed_module(self, token_count: int) -> MixedModule:
        for size, module in self.mixed_modules.items():
            if size >= token_count:
                return module

        raise RuntimeError(
            "Invalid token count."
        )
