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
        try:
            mdl = genai.GenerativeModel(model_name=model)
            parts = [system + "\n\n" + user]
            for b in images:
                parts.append(_to_blob(b))
            
            # Configure safety settings to be more permissive for business documents
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
            
            resp = mdl.generate_content(
                parts,
                generation_config={"temperature": temperature, "max_output_tokens": 4096},
                safety_settings=safety_settings
            )
        except Exception as api_error:
            log.error(f"Gemini API call failed with error: {api_error}")
            # Return fallback response for any API errors
            txt = '[{"伝票日付":"","借貸区分":"借方","科目名":"","金額":0,"摘要":"Gemini API呼び出し失敗【手動確認必要】"}]'
            return LLMResult(text=txt, tokens_in=0, tokens_out=0, cost_usd=0.0)
        
        # Handle different finish reasons
        if resp.candidates:
            candidate = resp.candidates[0]
            finish_reason = candidate.finish_reason
            
            # Convert finish_reason to string for robust comparison
            reason_str = str(finish_reason)
            reason_name = getattr(finish_reason, 'name', str(finish_reason))
            
            log.info(f"Gemini finish_reason: {reason_str} (name: {reason_name})")
            
            # Handle finish reasons robustly
            # First check for specific known cases by name
            if reason_name == 'STOP' or 'STOP' in reason_str or finish_reason == 1:
                # Normal completion
                try:
                    txt = resp.text or ""
                except Exception as e:
                    log.warning(f"Failed to get response text despite STOP finish_reason: {e}")
                    txt = ""
            
            elif reason_name == 'RECITATION' or 'RECITATION' in reason_str:
                # Blocked due to recitation/copyright concerns
                log.error(f"Gemini blocked response due to recitation/copyright concerns (finish_reason: {reason_str})")
                txt = '[{"伝票日付":"","借貸区分":"借方","科目名":"","金額":0,"摘要":"Gemini引用フィルタにより抽出中断【手動確認必要】"}]'
            
            elif reason_name == 'SAFETY' or 'SAFETY' in reason_str or finish_reason == 3:
                # Blocked due to safety filters
                log.error("Gemini blocked response due to safety filters")
                txt = '[{"伝票日付":"","借貸区分":"借方","科目名":"","金額":0,"摘要":"Gemini安全フィルタにより抽出中断【手動確認必要】"}]'
            
            elif reason_name == 'MAX_TOKENS' or 'MAX_TOKENS' in reason_str:
                # Response truncated due to token limit
                log.warning("Gemini response was truncated due to max_tokens limit")
                try:
                    txt = resp.text or ""
                except:
                    txt = ""
                if not txt:
                    txt = '[{"伝票日付":"","借貸区分":"借方","科目名":"","金額":0,"摘要":"Geminiトークン上限により抽出中断【手動確認必要】"}]'
            
            elif finish_reason == 2:
                # Special handling for finish_reason=2 which could be RECITATION in some API versions
                log.error(f"Gemini finish_reason=2 detected - likely recitation/copyright concern")
                txt = '[{"伝票日付":"","借貸区分":"借方","科目名":"","金額":0,"摘要":"Gemini引用フィルタにより抽出中断【手動確認必要】"}]'
            
            else:
                # Unknown or other finish reason
                log.error(f"Gemini returned unexpected finish_reason: {reason_str} (name: {reason_name})")
                try:
                    txt = resp.text or ""
                except:
                    txt = ""
                if not txt:
                    txt = '[{"伝票日付":"","借貸区分":"借方","科目名":"","金額":0,"摘要":"Gemini処理異常【手動確認必要】"}]'
        else:
            log.error("Gemini response contained no candidates")
            txt = '[{"伝票日付":"","借貸区分":"借方","科目名":"","金額":0,"摘要":"Gemini候補生成なし【手動確認必要】"}]'
        
        # Get usage metadata
        meta = getattr(resp, "usage_metadata", None)
        tin = getattr(meta, "prompt_token_count", 0) or 0
        tout = getattr(meta, "candidates_token_count", 0) or getattr(meta, "total_token_count", 0) or 0
        cost = tin * self.pr_in + tout * self.pr_out
        
        # Log detailed information for debugging
        if resp.candidates:
            finish_reason = resp.candidates[0].finish_reason
            log.info(f"Gemini response: finish_reason={finish_reason}, tokens_in={tin}, tokens_out={tout}, cost=${cost:.4f}")
        
        return LLMResult(
            text=txt,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost
        )