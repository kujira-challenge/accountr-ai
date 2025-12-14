#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ステップワイズPDF処理プロセッサ
1ステップ＝1分割処理で、Streamlitのrerunモデルに適合
Phase2対応: split単位の厳格なタイムアウト・例外の即時伝播
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from pdf_extractor import ProductionPDFExtractor
from config import config

logger = logging.getLogger(__name__)


class StepwiseProcessor:
    """1ステップずつ処理を進めるプロセッサ"""

    def __init__(self):
        """初期化"""
        self.extractor: Optional[ProductionPDFExtractor] = None
        logger.info("StepwiseProcessor initialized")

    def _ensure_extractor(self):
        """Extractorの遅延初期化"""
        if self.extractor is None:
            logger.info("Initializing ProductionPDFExtractor...")
            self.extractor = ProductionPDFExtractor(
                api_key=config.ANTHROPIC_API_KEY,
                use_mock=False
            )
            logger.info("ProductionPDFExtractor initialized successfully")

    def process_single_split(
        self,
        split_path: Path,
        split_index: int,
        total_splits: int,
        timeout_seconds: int = 120  # Phase2: split単位の厳格なタイムアウト
    ) -> Dict[str, Any]:
        """
        1つの分割を処理（1ステップ）

        Phase2修正:
        - split単位の厳格なタイムアウト（デフォルト120秒）
        - 例外を握り潰さず、詳細なエラー情報を返却
        - タイムアウト時は専用フラグを設定

        Args:
            split_path: 分割ファイルのパス
            split_index: 分割のインデックス（0始まり）
            total_splits: 総分割数
            timeout_seconds: タイムアウト時間（秒）

        Returns:
            Dict: 処理結果
                {
                    "success": bool,
                    "data": List[Dict],  # 抽出データ
                    "error": Optional[str],
                    "processing_time": float,
                    "split_info": {
                        "index": int,
                        "filename": str,
                        "pages": str  # "1-5" 形式
                    },
                    "entries_count": int,  # 抽出エントリ数
                    "timeout": bool  # タイムアウトフラグ
                }
        """
        logger.info(f"Processing split {split_index+1}/{total_splits}: {split_path.name} (timeout={timeout_seconds}s)")

        start_time = time.time()

        try:
            # Extractor初期化（遅延初期化）
            self._ensure_extractor()

            # ファイル名からページ範囲を抽出
            filename = split_path.stem
            page_range = "unknown"

            if '_pages_' in filename:
                try:
                    page_range = filename.split('_pages_')[1]
                except:
                    pass

            # ページ番号を抽出
            parts = page_range.split('-')
            if len(parts) == 2:
                try:
                    page_start = int(parts[0])
                    page_end = int(parts[1])
                except:
                    page_start = split_index + 1
                    page_end = split_index + 1
            else:
                page_start = split_index + 1
                page_end = split_index + 1

            logger.debug(f"Split page range: {page_start}-{page_end}")

            # Phase2: タイムアウト付きAPI呼び出し（ThreadPoolExecutorで厳格に制御）
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.extractor.extract_with_retry,
                    split_path,
                    page_start,
                    page_end
                )

                try:
                    extracted_data = future.result(timeout=timeout_seconds)
                except TimeoutError:
                    processing_time = time.time() - start_time
                    error_msg = f"処理がタイムアウトしました（{timeout_seconds}秒超過）"
                    logger.error(f"Split {split_index+1}/{total_splits} TIMEOUT: {processing_time:.1f}s > {timeout_seconds}s")

                    return {
                        "success": False,
                        "data": [],
                        "error": error_msg,
                        "processing_time": processing_time,
                        "split_info": {
                            "index": split_index,
                            "filename": split_path.name,
                            "pages": page_range
                        },
                        "entries_count": 0,
                        "timeout": True  # タイムアウトフラグ
                    }

            processing_time = time.time() - start_time
            entries_count = len(extracted_data) if extracted_data else 0

            logger.info(
                f"Split {split_index+1}/{total_splits} completed: "
                f"{entries_count} entries, {processing_time:.1f}s"
            )

            return {
                "success": True,
                "data": extracted_data,
                "error": None,
                "processing_time": processing_time,
                "split_info": {
                    "index": split_index,
                    "filename": split_path.name,
                    "pages": page_range
                },
                "entries_count": entries_count,
                "timeout": False
            }

        except Exception as e:
            # Phase2: 例外を握り潰さず、詳細なエラー情報をログ＋返却
            processing_time = time.time() - start_time
            logger.exception(f"Split {split_index+1}/{total_splits} FAILED: {e}")

            return {
                "success": False,
                "data": [],
                "error": str(e),
                "processing_time": processing_time,
                "split_info": {
                    "index": split_index,
                    "filename": split_path.name if split_path else "unknown",
                    "pages": "unknown"
                },
                "entries_count": 0,
                "timeout": False
            }

    def merge_results(self, split_results: list) -> Dict[str, Any]:
        """
        分割処理結果を統合

        Args:
            split_results: 分割処理結果のリスト

        Returns:
            Dict: 統合結果
                {
                    "success": bool,
                    "all_data": List[Dict],  # 全抽出データ
                    "total_entries": int,
                    "successful_splits": int,
                    "failed_splits": int,
                    "total_processing_time": float
                }
        """
        logger.info(f"Merging results from {len(split_results)} splits")

        all_data = []
        successful_splits = 0
        failed_splits = 0
        total_processing_time = 0.0

        for result in split_results:
            if result.get("success", False):
                successful_splits += 1
                if result.get("data"):
                    all_data.extend(result["data"])
            else:
                failed_splits += 1

            total_processing_time += result.get("processing_time", 0.0)

        total_entries = len(all_data)

        logger.info(
            f"Merge completed: {total_entries} entries, "
            f"{successful_splits} successful, {failed_splits} failed"
        )

        return {
            "success": failed_splits < len(split_results),  # 少なくとも1つ成功していればTrue
            "all_data": all_data,
            "total_entries": total_entries,
            "successful_splits": successful_splits,
            "failed_splits": failed_splits,
            "total_processing_time": total_processing_time
        }
