import os
import logging
from typing import List
from anthropic import Anthropic
from .base import LLMProvider, LLMResult

log = logging.getLogger(__name__)

class AnthropicProvider(LLMProvider):
    def __init__(self, pricing_in: float, pricing_out: float):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.pr_in, self.pr_out = pricing_in, pricing_out

    def generate(self, system: str, user: str, images: List[bytes], model: str, temperature: float = 0.0) -> LLMResult:
        content = [{"type": "text", "text": user}]
        for b in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b.decode("utf-8")
                }
            })
        
        resp = self.client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=4096
        )
        
        tin = getattr(resp.usage, "input_tokens", 0) or 0
        tout = getattr(resp.usage, "output_tokens", 0) or 0
        cost = tin * self.pr_in + tout * self.pr_out
        
        return LLMResult(
            text=resp.content[0].text if resp.content else "",
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost
        )