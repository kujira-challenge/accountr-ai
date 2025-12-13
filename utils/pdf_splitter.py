#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
適応的PDF分割ユーティリティ
PDFサイズに応じた最適な分割を実行
"""

import logging
from pathlib import Path
from typing import List, Tuple
from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


class AdaptivePDFSplitter:
    """PDFサイズに応じた適応的分割"""

    # 分割戦略の定義
    STRATEGY_SMALL = {"pages_per_split": 10, "max_pages": 30}      # 小規模: 10ページずつ
    STRATEGY_MEDIUM = {"pages_per_split": 5, "max_pages": 100}     # 中規模: 5ページずつ
    STRATEGY_LARGE = {"pages_per_split": 3, "max_pages": float('inf')}  # 大規模: 3ページずつ

    def __init__(self, temp_dir: Path):
        """
        初期化

        Args:
            temp_dir: 一時ファイル保存先ディレクトリ
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"AdaptivePDFSplitter initialized: temp_dir={self.temp_dir}")

    def determine_split_size(self, total_pages: int) -> int:
        """
        PDFサイズに応じた分割サイズを決定

        Args:
            total_pages: PDFの総ページ数

        Returns:
            int: ページ単位の分割サイズ
        """
        if total_pages <= self.STRATEGY_SMALL["max_pages"]:
            pages_per_split = self.STRATEGY_SMALL["pages_per_split"]
            strategy = "SMALL"
        elif total_pages <= self.STRATEGY_MEDIUM["max_pages"]:
            pages_per_split = self.STRATEGY_MEDIUM["pages_per_split"]
            strategy = "MEDIUM"
        else:
            pages_per_split = self.STRATEGY_LARGE["pages_per_split"]
            strategy = "LARGE"

        logger.info(f"Split strategy determined: {strategy} (total_pages={total_pages}, pages_per_split={pages_per_split})")
        return pages_per_split

    def split_pdf(self, pdf_path: Path) -> Tuple[List[Path], int, int]:
        """
        PDFを適応的に分割

        Args:
            pdf_path: 元PDFファイルのパス

        Returns:
            Tuple[List[Path], int, int]: (分割ファイルリスト, 総ページ数, 分割サイズ)

        Raises:
            ValueError: PDFの読み込みに失敗した場合
            Exception: 分割処理に失敗した場合
        """
        logger.info(f"Starting adaptive PDF split: {pdf_path.name}")

        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)

                if total_pages == 0:
                    raise ValueError("PDF contains no pages")

                # 分割サイズを決定
                pages_per_split = self.determine_split_size(total_pages)

                logger.info(f"Total pages: {total_pages}, pages_per_split: {pages_per_split}")

                split_files = []

                for start_page in range(0, total_pages, pages_per_split):
                    end_page = min(start_page + pages_per_split, total_pages)

                    try:
                        writer = PdfWriter()
                        for page_num in range(start_page, end_page):
                            writer.add_page(reader.pages[page_num])

                        # 分割ファイル名: original_split_001_pages_1-10.pdf
                        split_filename = f"{pdf_path.stem}_split_{len(split_files)+1:03d}_pages_{start_page+1}-{end_page}.pdf"
                        split_path = self.temp_dir / split_filename

                        with open(split_path, 'wb') as output_file:
                            writer.write(output_file)

                        split_files.append(split_path)
                        logger.debug(f"Created split {len(split_files)}: {split_filename}")

                    except Exception as e:
                        logger.error(f"Failed to create split {start_page+1}-{end_page}: {e}")
                        # 個別の分割失敗は続行
                        continue

                if not split_files:
                    raise Exception("No split files were created")

                logger.info(f"PDF split completed: {len(split_files)} splits created")
                return split_files, total_pages, pages_per_split

        except Exception as e:
            logger.error(f"PDF split failed: {pdf_path} - {e}")
            raise

    def cleanup_splits(self, split_files: List[Path]):
        """
        分割ファイルをクリーンアップ

        Args:
            split_files: 削除する分割ファイルのリスト
        """
        for split_file in split_files:
            try:
                if split_file.exists():
                    split_file.unlink()
                    logger.debug(f"Cleaned up split file: {split_file.name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {split_file}: {e}")

        logger.info(f"Cleanup completed: {len(split_files)} files")
