import os
import base64
import logging
from typing import List
import google.generativeai as genai
from google.generativeai import types as gtypes
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from .base import LLMProvider, LLMResult

log = logging.getLogger(__name__)

JSON_SYSTEM_GUARD = (
    "You are extracting structured ledger rows from USER-SUPPLIED images. "
    "Do NOT quote long texts. Return ONLY JSON (array of objects) with the specified schema."
)

SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

def _to_blob(b64jpeg):
    """Convert base64 JPEG (string or bytes) to Gemini blob format"""
    if isinstance(b64jpeg, str):
        raw = base64.b64decode(b64jpeg)
    else:
        raw = base64.b64decode(b64jpeg)
    return {"mime_type": "image/jpeg", "data": raw}

def _first_text(resp):
    """Safely extract text from Gemini response by traversing candidates → content → parts"""
    if not getattr(resp, "candidates", None):
        return None
    
    c0 = resp.candidates[0]
    content = getattr(c0, "content", None)
    parts = getattr(content, "parts", None) if content else None
    
    if parts:
        # Find first part with text
        for p in parts:
            text = getattr(p, "text", None)
            if text:
                return text
    return None

class GeminiProvider(LLMProvider):
    def __init__(self, pricing_in: float, pricing_out: float):
        # Try to import config and get API key through proper secrets management
        try:
            from config import Config
            config_instance = Config()
            api_key = config_instance.GOOGLE_API_KEY
        except:
            # Fallback to environment variable
            api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini provider. Set it in Streamlit secrets or environment variables.")
        
        genai.configure(api_key=api_key)
        self.pr_in, self.pr_out = pricing_in, pricing_out

    def _call(self, model: str, system: str, user: str, images_b64: List[bytes], *, json_mode: bool = False, max_out: int = 16384):
        """Internal call method with configurable JSON mode and token limits"""
        parts = [JSON_SYSTEM_GUARD + "\n\n" + system + "\n\n" + user]
        for b in images_b64:
            parts.append(_to_blob(b))

        # Generation configuration（空返し対策強化＋大量仕訳データ対応）
        gen_cfg = {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": max_out  # デフォルト16384に増量（大量JSON配列対応）
        }
        
        # Advanced safety settings - maximally permissive for business document extraction
        
        if json_mode:
            gen_cfg["response_mime_type"] = "application/json"
            # Optional JSON schema enforcement (SDK version dependent)
            try:
                # Check if Schema is available in the current SDK version
                if hasattr(gtypes, 'Schema') and hasattr(gtypes, 'Type'):
                    gen_cfg["response_schema"] = gtypes.Schema(
                        type=gtypes.Type.ARRAY,
                        items=gtypes.Schema(
                            type=gtypes.Type.OBJECT,
                            properties={
                                "伝票日付": gtypes.Schema(type=gtypes.Type.STRING),
                                "借貸区分": gtypes.Schema(type=gtypes.Type.STRING),
                                "科目名":   gtypes.Schema(type=gtypes.Type.STRING),
                                "金額":     gtypes.Schema(type=gtypes.Type.INTEGER),
                                "摘要":     gtypes.Schema(type=gtypes.Type.STRING),
                            },
                            required=["伝票日付", "借貸区分", "科目名", "金額", "摘要"]
                        )
                    )
                    log.info("Gemini JSON schema enforcement enabled")
                else:
                    log.info("Gemini JSON schema not available in this SDK version, using MIME type only")
            except Exception as schema_error:
                log.warning(f"Gemini JSON schema setup failed: {schema_error} - using MIME type only")

        mdl = genai.GenerativeModel(model_name=model, safety_settings=SAFETY_SETTINGS)
        
        # Add request timeout and retry logic（空返し対策強化＋大量データ対応）
        import time
        max_retries = 3  # 3回リトライ（504タイムアウト対策）
        base_timeout = 120  # seconds（504タイムアウト対策：60秒→120秒に延長）

        for attempt in range(max_retries):
            try:
                # Set reasonable timeout for API call
                resp = mdl.generate_content(parts, generation_config=gen_cfg, request_options={"timeout": base_timeout})

                # 空返しチェック（finish_reason=2等）
                text_content = _first_text(resp)
                if not text_content or not text_content.strip():
                    finish_reason = getattr(getattr(resp, "candidates", [{}])[0], "finish_reason", "unknown")
                    raise RuntimeError(f"Gemini returned empty text (finish_reason={finish_reason})")

                break  # Success, exit retry loop
            except Exception as call_error:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    raise call_error
                else:
                    # 指数バックオフ（1秒→2秒→4秒）
                    backoff_time = 2 ** attempt
                    log.warning(f"Gemini API call attempt {attempt + 1} failed: {call_error}, retrying in {backoff_time}s")
                    time.sleep(backoff_time)

        # Extract usage metadata (handle missing metadata gracefully)
        meta = getattr(resp, "usage_metadata", None)
        tin = getattr(meta, "prompt_token_count", 0) or 0
        tout = getattr(meta, "candidates_token_count", 0) or getattr(meta, "total_token_count", 0) or 0
        
        if tin == 0 and tout == 0:
            log.warning("Gemini usage metadata missing - cost calculation may be inaccurate")

        return resp, tin, tout

    def generate(self, system: str, user: str, images: List, model: str, temperature: float = 0.0) -> LLMResult:
        """Generate with comprehensive robustness and multi-step fallback"""
        total_tin, total_tout = 0, 0
        
        try:
            # Step 1: Normal mode attempt（早期フォールバック）
            log.info(f"Gemini Step 1: Normal mode with {model}")
            resp, tin, tout = self._call(model, system, user, images, json_mode=False, max_out=16384)
            text = _first_text(resp)
            total_tin, total_tout = tin, tout

            # 空返しは即座にエラー扱い（粘らない）
            if not text or not text.strip():
                finish_reason = getattr(getattr(resp, "candidates", [{}])[0], "finish_reason", "unknown")
                log.warning(f"Gemini Step 1 empty response: finish_reason={finish_reason}, failing fast for upstream fallback")
                raise RuntimeError(f"Gemini returned empty content (finish_reason={finish_reason})")

            # Success - calculate cost and return
            cost = total_tin * self.pr_in + total_tout * self.pr_out
            log.info(f"Gemini success: {len(text)} chars, tokens_in={total_tin}, tokens_out={total_tout}, cost=${cost:.4f}")

            return LLMResult(
                text=text,
                tokens_in=total_tin,
                tokens_out=total_tout,
                cost_usd=cost
            )

        except Exception as e:
            # Calculate cost even on failure for accurate tracking
            cost = total_tin * self.pr_in + total_tout * self.pr_out
            log.error(f"Gemini provider failed: {e} (cost so far: ${cost:.4f})")
            raise  # Re-raise for upstream fallback handling