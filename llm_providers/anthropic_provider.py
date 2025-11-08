import os
import base64
import logging
from typing import List
from io import BytesIO
from anthropic import Anthropic
from PIL import Image
from .base import LLMProvider, LLMResult

log = logging.getLogger(__name__)

# FORCE LOG OUTPUT TO CONFIRM CODE DEPLOYMENT
print("🔥 ANTHROPIC_PROVIDER.PY LOADED - NEW VERSION WITH DEBUG MARKERS 🔥")
log.error("🔥 ANTHROPIC_PROVIDER.PY LOADED - NEW VERSION WITH DEBUG MARKERS 🔥")

def _sniff_mime_from_b64(b64str: str) -> str:
    """Base64文字列から画像形式を自動判定してMIMEタイプを返す"""
    try:
        raw = base64.b64decode(b64str)
        with Image.open(BytesIO(raw)) as im:
            fmt = (im.format or "").lower()  # 'png', 'jpeg', 'webp', 'gif', ...
        if fmt == "jpg":
            fmt = "jpeg"
        if fmt not in {"png","jpeg","webp","gif"}:
            # 不明は jpeg にフォールバック
            fmt = "jpeg"
        return f"image/{fmt}"
    except Exception as e:
        log.warning(f"Failed to detect image format from base64: {e}, using jpeg fallback")
        return "image/jpeg"

def _image_part_from_b64(b64str: str) -> dict:
    """Base64文字列から自動MIME判定でAnthropicの画像パートを作成"""
    media_type = _sniff_mime_from_b64(b64str)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": b64str,
        },
    }

class AnthropicProvider(LLMProvider):
    def __init__(self, pricing_in: float, pricing_out: float):
        # Try to import config and get API key through proper secrets management
        try:
            from config import Config
            config_instance = Config()
            api_key = config_instance.ANTHROPIC_API_KEY
        except:
            # Fallback to environment variable
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider. Set it in Streamlit secrets or environment variables.")
        
        self.client = Anthropic(api_key=api_key)
        self.pr_in, self.pr_out = pricing_in, pricing_out

    def generate(self, system: str, user: str, images: List, model: str, temperature: float = 0.0) -> LLMResult:
        try:
            log.info(f"Anthropic provider called with {len(images)} images")
            for i, img in enumerate(images):
                log.info(f"Image {i}: type={type(img)}, length={len(str(img)) if img else 0}")

            content = []
            for i, b in enumerate(images):
                # Universal data conversion - handle any format
                try:
                    # Try to convert to string regardless of current type
                    if isinstance(b, bytes):
                        # It's bytes-like
                        log.error(f"[BYTES_PATH] Converting bytes to string for image {i}")
                        data = b.decode("utf-8")
                        log.info(f"Image {i}: Converted bytes to string (length={len(data)})")
                    elif isinstance(b, str):
                        # It's already a string
                        data = b
                        log.info(f"Image {i}: Using as string (length={len(data)})")
                    else:
                        # Try to convert to string
                        data = str(b)
                        log.info(f"Image {i}: Converted to string (length={len(data)}), type was: {type(b)}")

                    content.append(_image_part_from_b64(data))

                except Exception as conversion_error:
                    log.error(f"[CONVERSION_ERROR] Image {i}: Failed to convert to string: {conversion_error}, type: {type(b)}")
                    # Skip this image rather than failing completely
                    continue

            content.append({"type": "text", "text": user})
            
            resp = self.client.messages.create(
                model=model,
                system=system,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=16384  # Claude最適化: 適切な出力上限（コスト効率重視）
            )
            
            # Safe token count extraction
            tin = getattr(resp.usage, "input_tokens", 0) or 0
            tout = getattr(resp.usage, "output_tokens", 0) or 0
            cost = tin * self.pr_in + tout * self.pr_out
            
            # Safe text extraction with fallback
            try:
                text = resp.content[0].text if resp.content and len(resp.content) > 0 else ""
                if not text or len(text.strip()) < 10:  # Check for essentially empty response
                    log.warning("Anthropic returned empty or very short response")
                    text = '[{"伝票日付":"","借貸区分":"借方","科目名":"抽出失敗","金額":100,"摘要":"Anthropicレスポンス抽出失敗【手動確認必要】"}]'
            except (IndexError, AttributeError) as e:
                log.warning(f"Failed to extract text from Anthropic response: {e}")
                text = '[{"伝票日付":"","借貸区分":"借方","科目名":"抽出失敗","金額":1,"摘要":"Anthropicレスポンス抽出失敗【手動確認必要】"}]'
            
            return LLMResult(
                text=text,
                tokens_in=tin,
                tokens_out=tout,
                cost_usd=cost
            )
            
        except Exception as e:
            import traceback
            log.error(f"[DEBUG_MARKER] Anthropic API call failed: {e}")
            log.error(f"Full traceback: {traceback.format_exc()}")
            # Return fallback response with valid amount for validation
            fallback_text = '[{"伝票日付":"","借貸区分":"借方","科目名":"API失敗","金額":100,"摘要":"Anthropic API呼び出し失敗【手動確認必要】"}]'
            log.error(f"Returning fallback response: {fallback_text}")
            return LLMResult(
                text=fallback_text,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0
            )