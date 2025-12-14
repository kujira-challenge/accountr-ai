#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-Based PDF Processor
完全ノンブロッキング - ThreadPoolExecutor廃止、1 rerun = 1フェーズ厳守
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import json

from utils.split_phases import SplitPhase, SplitProcessingState
from llm_providers.factory import build as build_llm
from config import config

logger = logging.getLogger(__name__)


class PhaseBasedProcessor:
    """
    完全フェーズベースのプロセッサ

    重要な設計原則:
    - ThreadPoolExecutor / multiprocessing / future.result(timeout) を一切使わない
    - 各フェーズは必ずメインスレッドで実行
    - 各フェーズ完了後は必ず session_state に保存して st.rerun()
    - 1 rerun = 1フェーズを厳守
    """

    def __init__(self):
        """初期化"""
        self.llm_client: Optional[Any] = None
        logger.info("PhaseBasedProcessor initialized")

    def _ensure_llm_client(self):
        """LLMクライアントの遅延初期化"""
        if self.llm_client is None:
            logger.info("Initializing LLM client...")
            try:
                # config.yamlから設定読み込み
                import yaml
                with open("config.yaml", "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)

                provider = cfg.get("llm", {}).get("provider", "gemini")
                model = cfg.get("llm", {}).get("model", "gemini-2.5-flash")
                temperature = cfg.get("llm", {}).get("temperature", 0.0)
                pricing = cfg.get("pricing", {})

                logger.info(f"Building LLM client: {provider}/{model}")
                self.llm_client = build_llm(
                    provider=provider,
                    model=model,
                    pricing=pricing
                )
                self.llm_model = model
                self.llm_temperature = temperature
                logger.info("LLM client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                raise

    def process_phase(
        self,
        split_state: SplitProcessingState,
        split_path: Path,
        total_splits: int
    ) -> Dict[str, Any]:
        """
        1つのフェーズを処理（1 rerun = 1 phase）

        Args:
            split_state: Split処理の現在の状態
            split_path: 分割PDFファイルのパス
            total_splits: 総分割数

        Returns:
            Dict: フェーズ処理結果
                {
                    "phase_complete": bool,  # このフェーズが完了したか
                    "split_complete": bool,  # Split全体が完了したか
                    "success": bool,
                    "error": Optional[str],
                    "next_action": str  # "continue" | "rerun" | "next_split"
                }
        """
        logger.info(
            f"Processing split {split_state.split_index+1}/{total_splits} "
            f"phase={split_state.phase.value}"
        )

        try:
            # フェーズごとの処理を実行
            if split_state.phase == SplitPhase.GEMINI_CALL:
                return self._phase_gemini_call(split_state, split_path)

            elif split_state.phase == SplitPhase.JSON_PARSE:
                return self._phase_json_parse(split_state)

            elif split_state.phase == SplitPhase.POSTPROCESS:
                return self._phase_postprocess(split_state)

            elif split_state.phase == SplitPhase.VALIDATION:
                return self._phase_validation(split_state)

            elif split_state.phase == SplitPhase.COMPLETED:
                return {
                    "phase_complete": True,
                    "split_complete": True,
                    "success": True,
                    "error": None,
                    "next_action": "next_split"
                }

            elif split_state.phase == SplitPhase.FAILED:
                return {
                    "phase_complete": True,
                    "split_complete": True,
                    "success": False,
                    "error": split_state.error,
                    "next_action": "next_split"
                }

            else:
                raise ValueError(f"Unknown phase: {split_state.phase}")

        except Exception as e:
            logger.exception(f"Phase {split_state.phase.value} failed: {e}")
            split_state.mark_failed(f"Phase {split_state.phase.value} error: {str(e)}")
            return {
                "phase_complete": True,
                "split_complete": True,
                "success": False,
                "error": str(e),
                "next_action": "next_split"
            }

    def _phase_gemini_call(
        self,
        split_state: SplitProcessingState,
        split_path: Path
    ) -> Dict[str, Any]:
        """
        GEMINI_CALLフェーズ: Gemini APIを呼び出す

        このフェーズだけで処理が重い場合でも、
        メインスレッドで実行されるため、制御は必ず戻る。
        """
        logger.info(f"Phase GEMINI_CALL: Calling Gemini API for {split_path.name}")

        try:
            # LLMクライアント初期化
            self._ensure_llm_client()

            # PDFを画像に変換（既存のpdf_to_images_base64メソッドを使用）
            start_time = time.time()
            logger.debug(f"Converting PDF to images: {split_path.name}")

            images_b64 = self._pdf_to_images_base64(split_path)
            if not images_b64:
                raise ValueError(f"No images extracted from {split_path.name}")

            logger.debug(f"Converted {len(images_b64)} pages to images")

            # プロンプト作成
            system_prompt = self._get_system_prompt()
            user_prompt = f"ファイル名: {split_path.name}, ページ: {split_state.page_start}-{split_state.page_end}\n"
            for page_idx in range(1, len(images_b64) + 1):
                user_prompt += f"ページ{page_idx}：左=借方｜中央=摘要｜右=貸方（列を取り違えないでください）\n"

            # Gemini API呼び出し（ここがブロックする可能性があるが、メインスレッドなので制御は戻る）
            logger.debug(f"Sending request to Gemini API ({len(images_b64)} images)")

            # LLMに処理依頼
            result = self.llm_client.generate(
                system=system_prompt,
                user=user_prompt,
                images=images_b64,
                model=self.llm_model,
                temperature=self.llm_temperature
            )

            elapsed = time.time() - start_time
            logger.info(f"Gemini API call completed in {elapsed:.1f}s")

            # レスポンスを保存
            split_state.gemini_response = {
                "text": result.text,
                "cost_usd": result.cost_usd
            }

            # 次のフェーズへ
            split_state.advance_phase()

            return {
                "phase_complete": True,
                "split_complete": False,
                "success": True,
                "error": None,
                "next_action": "rerun"
            }

        except Exception as e:
            logger.error(f"GEMINI_CALL phase failed: {e}")
            split_state.mark_failed(f"Gemini API error: {str(e)}")
            return {
                "phase_complete": True,
                "split_complete": True,
                "success": False,
                "error": str(e),
                "next_action": "next_split"
            }

    def _pdf_to_images_base64(self, pdf_path: Path) -> List[str]:
        """PDFを画像に変換してBase64エンコード"""
        import fitz  # PyMuPDF
        from PIL import Image
        import io
        import base64

        images_b64 = []
        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page to image (DPI=150 for balance between quality and file size)
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Convert to JPEG for smaller file size
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                buffer.seek(0)

                # Encode to base64
                img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
                images_b64.append(img_b64)

            doc.close()
            return images_b64

        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise

    def _get_system_prompt(self) -> str:
        """Gemini最適化版プロンプト"""
        return """あなたは経理補助AIです。伝票画像から複式簿記の仕訳をJSON形式で出力してください。

【複式簿記原則】
- 1取引＝借方1本以上＋貸方1本以上
- 借方・貸方の合計金額は必ず一致
- 単一側（借方のみ／貸方のみ）の出力は禁止
- 同一取引の借方・貸方は、それぞれ別のオブジェクトとして出力
- 説明文・コードブロック禁止

【出力形式】
JSONオブジェクト配列のみ。5キー必須: ["伝票日付","借貸区分","科目名","金額","摘要"]
任意で "仕訳No" を付けてもよい。仕訳Noは同一伝票内の全行で同一値とし、ページ内出現順に 001,002,... と3桁ゼロ埋めで採番。

【形式例】
[
  {"仕訳No":"001","伝票日付":"2025/2/14","借貸区分":"借方","科目名":"退居敷金","金額":55000,"摘要":"ルベール鷺宮102号 白坂安純様 敷金返還精算"},
  {"仕訳No":"001","伝票日付":"2025/2/14","借貸区分":"貸方","科目名":"現金（西京普通）","金額":55000,"摘要":"ルベール鷺宮102号 白坂安純様 敷金返還精算"}
]

【抽出ルール】
- 左列=借方、右列=貸方
- 各取引は借方・貸方ペアで出力（枠ごとの借方合計と貸方合計を一致させる）
- 不明科目は空文字または"不明"
- 個人名は必要最小限で記載
- OCR曖昧な場合は摘要末尾に【OCR注意】
- 抽出不能時も1要素返す"""

    def _phase_json_parse(self, split_state: SplitProcessingState) -> Dict[str, Any]:
        """
        JSON_PARSEフェーズ: Geminiのレスポンスから JSON をパース
        """
        logger.info(f"Phase JSON_PARSE: Parsing Gemini response")

        try:
            response = split_state.gemini_response
            if not response:
                raise ValueError("No Gemini response available")

            # レスポンスからJSONを抽出
            response_text = response.get("text", "")
            if not response_text:
                raise ValueError("Empty response from Gemini")

            # JSON抽出ガードを使用してパース
            from utils.json_guard import parse_5cols_json
            parsed_data = parse_5cols_json(response_text)

            logger.info(f"JSON parsed successfully: {len(parsed_data)} entries")

            # 保存
            split_state.parsed_json = parsed_data

            # 次のフェーズへ
            split_state.advance_phase()

            return {
                "phase_complete": True,
                "split_complete": False,
                "success": True,
                "error": None,
                "next_action": "rerun"
            }

        except Exception as e:
            logger.error(f"JSON_PARSE phase failed: {e}")
            split_state.mark_failed(f"JSON parse error: {str(e)}")
            return {
                "phase_complete": True,
                "split_complete": True,
                "success": False,
                "error": str(e),
                "next_action": "next_split"
            }

    def _phase_postprocess(self, split_state: SplitProcessingState) -> Dict[str, Any]:
        """
        POSTPROCESSフェーズ: データの後処理（前段整形）
        """
        logger.info(f"Phase POSTPROCESS: Post-processing data")

        try:
            parsed_data = split_state.parsed_json
            if not parsed_data:
                logger.warning("No parsed JSON available for post-processing")
                split_state.processed_data = []
            else:
                # 5カラムの前段整形（コード列を見ない段階での重複・形状整理）
                from utils.reconcile_entries import reconcile_entries
                processed_data, _ = reconcile_entries(parsed_data, sum_tolerance=0, return_metrics=True)

                logger.info(f"Post-processing completed: {len(processed_data)} entries (from {len(parsed_data)})")

                # 保存
                split_state.processed_data = processed_data

            # 次のフェーズへ
            split_state.advance_phase()

            return {
                "phase_complete": True,
                "split_complete": False,
                "success": True,
                "error": None,
                "next_action": "rerun"
            }

        except Exception as e:
            logger.error(f"POSTPROCESS phase failed: {e}")
            split_state.mark_failed(f"Post-process error: {str(e)}")
            return {
                "phase_complete": True,
                "split_complete": True,
                "success": False,
                "error": str(e),
                "next_action": "next_split"
            }

    def _phase_validation(self, split_state: SplitProcessingState) -> Dict[str, Any]:
        """
        VALIDATIONフェーズ: データの検証（貸借ペア保証と金額バリデーション）
        """
        logger.info(f"Phase VALIDATION: Validating data")

        try:
            processed_data = split_state.processed_data
            if not processed_data:
                logger.warning("No processed data to validate")
                split_state.validated_data = []
            else:
                # 後処理: 貸借ペア保証と金額バリデーション
                from utils.postprocess import enforce_debit_credit_pairs, validate_amounts
                paired_entries = enforce_debit_credit_pairs(processed_data)
                validated_data, _ = validate_amounts(paired_entries)

                logger.info(
                    f"Validation completed: {len(validated_data)}/{len(processed_data)} entries valid"
                )

                split_state.validated_data = validated_data

            # 次のフェーズへ（COMPLETED）
            split_state.advance_phase()

            return {
                "phase_complete": True,
                "split_complete": True,  # VALIDATIONが最後のフェーズ
                "success": True,
                "error": None,
                "next_action": "next_split"
            }

        except Exception as e:
            logger.error(f"VALIDATION phase failed: {e}")
            split_state.mark_failed(f"Validation error: {str(e)}")
            return {
                "phase_complete": True,
                "split_complete": True,
                "success": False,
                "error": str(e),
                "next_action": "next_split"
            }

    def merge_results(self, split_states: List[SplitProcessingState]) -> Dict[str, Any]:
        """
        分割処理結果を統合

        Args:
            split_states: 分割処理状態のリスト

        Returns:
            Dict: 統合結果
        """
        logger.info(f"Merging results from {len(split_states)} splits")

        all_data = []
        successful_splits = 0
        failed_splits = 0

        for state in split_states:
            if state.phase == SplitPhase.COMPLETED:
                successful_splits += 1
                data = state.get_final_data()
                if data:
                    all_data.extend(data)
            else:
                failed_splits += 1

        total_entries = len(all_data)

        logger.info(
            f"Merge completed: {total_entries} entries, "
            f"{successful_splits} successful, {failed_splits} failed"
        )

        return {
            "success": successful_splits > 0,
            "all_data": all_data,
            "total_entries": total_entries,
            "successful_splits": successful_splits,
            "failed_splits": failed_splits,
        }
