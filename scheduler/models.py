from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass, field

@dataclass
class Request:
    future: Future
    prompt: list[int]
    t_arrival: float
    t_first: float = 0.0
    kv_len: int = 0
    budget: int = 0
    new_tokens: list[int] = field(default_factory=list)
    pages: set[int] = field(default_factory=set)
    prefilled: bool = False

    def requires_page(self, page_size: int) -> bool:
        return len(self.pages) * page_size == self.kv_len

    def uncached_token_count(self) -> int:
        return len(self.prompt) + len(self.new_tokens) - self.kv_len


@dataclass
class ComponentInitStats:
    seconds: float
    memory: int
    memory_allocated: int
    memory_reserved: int


@dataclass
class InitStats:
    total_seconds: float = 0.0
    model: ComponentInitStats | None = None
    kv_cache: ComponentInitStats | None = None
    piecewise: ComponentInitStats | None = None
    full: ComponentInitStats | None = None


@dataclass
class RunStats:
    decode_steps: Counter = field(default_factory=Counter)
    full_mixed_steps: Counter = field(default_factory=Counter)
    piecewise_mixed_steps: Counter = field(default_factory=Counter)
    batch_sizes: Counter = field(default_factory=Counter)
    tokens: int = 0
    bucket_tokens: int = 0
    preemptions: int = 0
    recomputed_tokens: int = 0


@dataclass
class Tracker:
    init: InitStats = field(default_factory=InitStats)
    run: RunStats = field(default_factory=RunStats)
