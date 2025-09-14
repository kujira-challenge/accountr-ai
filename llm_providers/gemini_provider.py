import os
import base64
import logging
from typing import List
import google.generativeai as genai
from google.generativeai import types as gtypes
from .base import LLMProvider, LLMResult

log = logging.getLogger(__name__)

JSON_SYSTEM_GUARD = (
    "You are extracting structured ledger rows from USER-SUPPLIED images. "
    "Do NOT quote long texts. Return ONLY JSON (array of objects) with the specified schema."
)

def _to_blob(b64jpeg: bytes):
    """Convert base64 JPEG bytes to Gemini blob format"""
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
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.pr_in, self.pr_out = pricing_in, pricing_out

    def _call(self, model: str, system: str, user: str, images_b64: List[bytes], *, json_mode: bool = False, max_out: int = 2048):
        """Internal call method with configurable JSON mode and token limits"""
        parts = [JSON_SYSTEM_GUARD + "\n\n" + system + "\n\n" + user]
        for b in images_b64:
            parts.append(_to_blob(b))

        # Generation configuration
        gen_cfg = {
            "temperature": 0.0, 
            "max_output_tokens": max_out
        }
        
        # Advanced safety settings - maximally permissive for business document extraction
        safety = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
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

        mdl = genai.GenerativeModel(model_name=model, safety_settings=safety)
        
        # Add request timeout and retry logic
        import time
        max_retries = 2
        base_timeout = 30  # seconds
        
        for attempt in range(max_retries):
            try:
                # Set reasonable timeout for API call
                resp = mdl.generate_content(parts, generation_config=gen_cfg, request_options={"timeout": base_timeout * (attempt + 1)})
                break  # Success, exit retry loop
            except Exception as call_error:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    raise call_error
                else:
                    log.warning(f"Gemini API call attempt {attempt + 1} failed: {call_error}, retrying in {2 ** attempt}s")
                    time.sleep(2 ** attempt)  # Exponential backoff

        # Extract usage metadata (handle missing metadata gracefully)
        meta = getattr(resp, "usage_metadata", None)
        tin = getattr(meta, "prompt_token_count", 0) or 0
        tout = getattr(meta, "candidates_token_count", 0) or getattr(meta, "total_token_count", 0) or 0
        
        if tin == 0 and tout == 0:
            log.warning("Gemini usage metadata missing - cost calculation may be inaccurate")

        return resp, tin, tout

    def generate(self, system: str, user: str, images: List[bytes], model: str, temperature: float = 0.0) -> LLMResult:
        """Generate with comprehensive robustness and multi-step fallback"""
        total_tin, total_tout = 0, 0
        
        try:
            # Step 1: Normal mode attempt
            log.info(f"Gemini Step 1: Normal mode with {model}")
            resp, tin, tout = self._call(model, system, user, images, json_mode=False, max_out=2048)
            text = _first_text(resp)
            total_tin, total_tout = tin, tout

            # Step 2: If empty parts, try JSON mode with increased tokens
            if not text:
                finish_reason = getattr(getattr(resp, "candidates", [{}])[0], "finish_reason", "unknown")
                log.warning(f"Gemini empty parts: finish_reason={finish_reason}, retrying with JSON mode")
                
                try:
                    resp, tin2, tout2 = self._call(model, system, user, images, json_mode=True, max_out=4096)
                    text = _first_text(resp)
                    total_tin, total_tout = total_tin + tin2, total_tout + tout2
                    log.info(f"Gemini JSON mode retry completed: text_length={len(text or '')}")
                except Exception as json_retry_error:
                    log.error(f"Gemini JSON mode retry failed: {json_retry_error}")
                    # Don't re-raise here, continue to Step 3 check

            # Step 3: If still empty, raise explicit exception for upstream fallback
            if not text:
                final_reason = getattr(getattr(resp, "candidates", [{}])[0], "finish_reason", "unknown")
                log.error(f"Gemini completely failed after all retries: finish_reason={final_reason}")
                raise RuntimeError(f"Gemini returned no content parts after retries (finish_reason={final_reason})")

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