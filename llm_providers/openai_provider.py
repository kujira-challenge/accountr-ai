import os
import logging
from typing import List
from openai import OpenAI
from .base import LLMProvider, LLMResult

log = logging.getLogger(__name__)

class OpenAIProvider(LLMProvider):
    def __init__(self, pricing_in: float, pricing_out: float):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.pr_in, self.pr_out = pricing_in, pricing_out

    def generate(self, system: str, user: str, images: List[bytes], model: str, temperature: float = 0.0) -> LLMResult:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}]}
        ]
        
        # Add images to user message
        for b in images:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b.decode('utf-8')}"
                }
            })
        
        # For GPT-5, use new parameters if available
        generation_config = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096
        }
        
        # Add GPT-5 specific parameters if model is GPT-5
        if model.startswith("gpt-5"):
            generation_config.update({
                "reasoning_effort": "medium",  # New GPT-5 parameter
                "response_format": {"type": "text"}  # Use Responses API format
            })
        
        resp = self.client.chat.completions.create(**generation_config)
        
        tin = getattr(resp.usage, "prompt_tokens", 0) or 0
        tout = getattr(resp.usage, "completion_tokens", 0) or 0
        cost = tin * self.pr_in + tout * self.pr_out
        
        return LLMResult(
            text=resp.choices[0].message.content if resp.choices else "",
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost
        )