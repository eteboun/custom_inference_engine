from kv_cache.storage import KVStorage
from scheduler.decode_module import DecodeModule
from scheduler.mixed_module import MixedModule

class Bucket:
    def __init__(self, model, kv_cache: KVStorage, batch_size: int):
        self.batch_size = batch_size

        self.decode_module = DecodeModule(
            model=model,
            kv_cache=kv_cache,
            batch_size=batch_size
        )

        self.mixed_module = MixedModule(
            model=model,
            kv_cache=kv_cache,
            batch_size=batch_size
        )
