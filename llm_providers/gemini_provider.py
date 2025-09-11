import os
import logging
import base64
from typing import List
import google.generativeai as genai
from .base import LLMProvider, LLMResult

log = logging.getLogger(__name__)

def _to_blob(b64jpeg: bytes):
    raw = base64.b64decode(b64jpeg)
    return {"mime_type": "image/jpeg", "data": raw}

class GeminiProvider(LLMProvider):
    def __init__(self, pricing_in: float, pricing_out: float):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.pr_in, self.pr_out = pricing_in, pricing_out

    def generate(self, system: str, user: str, images: List[bytes], model: str, temperature: float = 0.0) -> LLMResult:
        mdl = genai.GenerativeModel(model_name=model)
        parts = [system + "\n\n" + user]
        for b in images:
            parts.append(_to_blob(b))
        
        resp = mdl.generate_content(
            parts,
            generation_config={"temperature": temperature, "max_output_tokens": 4096},
        )
        
        txt = resp.text or ""
        meta = getattr(resp, "usage_metadata", None)
        tin = getattr(meta, "prompt_token_count", 0) or 0
        tout = getattr(meta, "candidates_token_count", 0) or getattr(meta, "total_token_count", 0) or 0
        cost = tin * self.pr_in + tout * self.pr_out
        
        return LLMResult(
            text=txt,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost
        )