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
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": user}]}
            ]
            
            # Add images to user message
            for i, b in enumerate(images):
                try:
                    # Handle both bytes and string inputs
                    if isinstance(b, bytes):
                        data = b.decode('utf-8')
                        log.debug(f"Image {i}: Converted bytes to string (length={len(data)})")
                    elif isinstance(b, str):
                        data = b
                        log.debug(f"Image {i}: Using as string (length={len(data)})")
                    else:
                        data = str(b)
                        log.debug(f"Image {i}: Converted to string (length={len(data)}), type was: {type(b)}")

                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{data}"
                        }
                    })
                except Exception as conversion_error:
                    log.error(f"Image {i}: Failed to convert: {conversion_error}, type: {type(b)}")
                    continue
            
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
            
            # Safe token count extraction
            tin = getattr(resp.usage, "prompt_tokens", 0) or 0
            tout = getattr(resp.usage, "completion_tokens", 0) or 0
            cost = tin * self.pr_in + tout * self.pr_out
            
            # Safe text extraction with fallback
            try:
                text = resp.choices[0].message.content if resp.choices and len(resp.choices) > 0 and resp.choices[0].message else ""
            except (IndexError, AttributeError) as e:
                log.warning(f"Failed to extract text from OpenAI response: {e}")
                text = '[{"伝票日付":"","借貸区分":"借方","科目名":"OpenAI失敗","金額":100,"摘要":"OpenAIレスポンス抽出失敗【手動確認必要】"}]'
            
            return LLMResult(
                text=text or "",
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost
            )
            
        except Exception as e:
            log.error(f"OpenAI API call failed: {e}")
            # Return fallback response for any API errors
            return LLMResult(
                text='[{"伝票日付":"","借貸区分":"借方","科目名":"OpenAI失敗","金額":100,"摘要":"OpenAI API呼び出し失敗【手動確認必要】"}]',
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0
            )