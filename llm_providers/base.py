from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float

class LLMProvider:
    def generate(self, system: str, user: str, images: List, model: str, temperature: float = 0.0) -> LLMResult:
        raise NotImplementedError