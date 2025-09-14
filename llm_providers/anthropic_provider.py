import os
import logging
from typing import List
from anthropic import Anthropic
from .base import LLMProvider, LLMResult

log = logging.getLogger(__name__)

class AnthropicProvider(LLMProvider):
    def __init__(self, pricing_in: float, pricing_out: float):
        # Try to import config and get API key through proper secrets management
        try:
            from config import config
            api_key = config.ANTHROPIC_API_KEY
        except:
            # Fallback to environment variable
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider. Set it in Streamlit secrets or environment variables.")
        
        self.client = Anthropic(api_key=api_key)
        self.pr_in, self.pr_out = pricing_in, pricing_out

    def generate(self, system: str, user: str, images: List[bytes], model: str, temperature: float = 0.0) -> LLMResult:
        try:
            content = [{"type": "text", "text": user}]
            for b in images:
                # Handle both bytes and str format for base64 data
                if isinstance(b, bytes):
                    data = b.decode("utf-8")
                else:
                    data = b
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": data
                    }
                })
            
            resp = self.client.messages.create(
                model=model,
                system=system,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=4096
            )
            
            # Safe token count extraction
            tin = getattr(resp.usage, "input_tokens", 0) or 0
            tout = getattr(resp.usage, "output_tokens", 0) or 0
            cost = tin * self.pr_in + tout * self.pr_out
            
            # Safe text extraction with fallback
            try:
                text = resp.content[0].text if resp.content and len(resp.content) > 0 else ""
            except (IndexError, AttributeError) as e:
                log.warning(f"Failed to extract text from Anthropic response: {e}")
                text = '[{"伝票日付":"","借貸区分":"借方","科目名":"","金額":0,"摘要":"Anthropicレスポンス抽出失敗【手動確認必要】"}]'
            
            return LLMResult(
                text=text,
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost
            )
            
        except Exception as e:
            log.error(f"Anthropic API call failed: {e}")
            # Return fallback response for any API errors
            return LLMResult(
                text='[{"伝票日付":"","借貸区分":"借方","科目名":"","金額":0,"摘要":"Anthropic API呼び出し失敗【手動確認必要】"}]',
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0
            )