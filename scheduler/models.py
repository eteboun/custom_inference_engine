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