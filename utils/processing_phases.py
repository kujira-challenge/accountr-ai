#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
処理フェーズ定義と状態管理
Streamlitのステートマシン型処理フロー用
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import time
import logging

logger = logging.getLogger(__name__)


class ProcessingPhase(Enum):
    """処理フェーズの定義"""
    IDLE = "idle"                    # 待機中
    SPLITTING = "splitting"          # PDF分割中
    PROCESSING = "processing"        # 分割単位で処理中
    MERGING = "merging"              # 結果統合中
    COMPLETED = "completed"          # 完了
    ERROR = "error"                  # エラー
    TIMEOUT = "timeout"              # タイムアウト


@dataclass
class ProcessingState:
    """処理状態の完全な定義"""

    # フェーズ管理
    phase: ProcessingPhase = ProcessingPhase.IDLE

    # PDF情報
    pdf_name: str = ""
    total_pages: int = 0

    # 分割情報
    split_files: List[str] = field(default_factory=list)  # 分割ファイルパスのリスト
    total_splits: int = 0
    pages_per_split: int = 0  # 分割サイズ

    # 処理進捗
    current_split_index: int = 0
    split_results: List[Dict[str, Any]] = field(default_factory=list)  # 各分割の処理結果

    # エラー・警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # タイムアウト管理
    start_time: float = 0.0
    timeout_seconds: int = 900  # 15分

    # 最終結果
    final_df: Optional[Any] = None  # pandas DataFrameだがserializable対応のためAny
    final_csv_bytes: Optional[bytes] = None
    processing_info: Dict[str, Any] = field(default_factory=dict)

    def is_timeout(self) -> bool:
        """
        タイムアウトチェック

        Returns:
            bool: タイムアウトしている場合True
        """
        if self.start_time == 0.0:
            return False
        elapsed = time.time() - self.start_time
        is_timeout = elapsed >= self.timeout_seconds

        if is_timeout:
            logger.warning(f"タイムアウト検出: {elapsed:.1f}秒 / {self.timeout_seconds}秒")

        return is_timeout

    def get_elapsed(self) -> float:
        """
        経過時間を取得

        Returns:
            float: 経過時間（秒）
        """
        if self.start_time == 0.0:
            return 0.0
        return time.time() - self.start_time

    def get_elapsed_str(self) -> str:
        """
        経過時間を文字列で取得

        Returns:
            str: "X分Y秒" 形式の文字列
        """
        elapsed = self.get_elapsed()
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}分{seconds}秒"

    def get_progress_percentage(self) -> float:
        """
        進捗率を取得（0.0-1.0）

        Returns:
            float: 進捗率
        """
        if self.total_splits == 0:
            return 0.0
        return min(1.0, self.current_split_index / self.total_splits)

    def get_successful_splits_count(self) -> int:
        """
        成功した分割数を取得

        Returns:
            int: 成功した分割数
        """
        return sum(1 for r in self.split_results if r.get("success", False))

    def get_failed_splits_count(self) -> int:
        """
        失敗した分割数を取得

        Returns:
            int: 失敗した分割数
        """
        return sum(1 for r in self.split_results if not r.get("success", True))

    def get_total_extracted_entries(self) -> int:
        """
        抽出された総エントリ数を取得

        Returns:
            int: 総エントリ数
        """
        total = 0
        for result in self.split_results:
            if result.get("success", False) and result.get("data"):
                total += len(result["data"])
        return total

    def reset(self):
        """状態をリセット"""
        self.phase = ProcessingPhase.IDLE
        self.pdf_name = ""
        self.total_pages = 0
        self.split_files = []
        self.total_splits = 0
        self.pages_per_split = 0
        self.current_split_index = 0
        self.split_results = []
        self.errors = []
        self.warnings = []
        self.start_time = 0.0
        self.final_df = None
        self.final_csv_bytes = None
        self.processing_info = {}

        logger.info("Processing state reset")
