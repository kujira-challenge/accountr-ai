#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF関連ユーティリティ
"""

import io
import logging
from PyPDF2 import PdfReader
from typing import Union
from pathlib import Path

logger = logging.getLogger(__name__)


def get_pdf_page_count(pdf_source: Union[bytes, io.BytesIO, Path]) -> int:
    """
    PDFファイルのページ数を取得

    Args:
        pdf_source: PDFのソース（bytes, BytesIO, またはPath）

    Returns:
        int: ページ数（エラー時は0）
    """
    try:
        # バイトデータの場合
        if isinstance(pdf_source, bytes):
            pdf_file = io.BytesIO(pdf_source)
        # BytesIOの場合
        elif isinstance(pdf_source, io.BytesIO):
            pdf_file = pdf_source
            pdf_file.seek(0)  # ポインタを先頭に戻す
        # Pathの場合
        elif isinstance(pdf_source, Path):
            with open(pdf_source, 'rb') as f:
                pdf_bytes = f.read()
            pdf_file = io.BytesIO(pdf_bytes)
        # Streamlit UploadedFileの場合（seekメソッドを持つ）
        elif hasattr(pdf_source, 'read') and hasattr(pdf_source, 'seek'):
            pdf_source.seek(0)
            pdf_bytes = pdf_source.read()
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_source.seek(0)  # 元に戻す
        else:
            logger.error(f"Unsupported pdf_source type: {type(pdf_source)}")
            return 0

        # PDFReaderでページ数を取得
        reader = PdfReader(pdf_file)
        page_count = len(reader.pages)

        logger.info(f"PDF page count: {page_count}")
        return page_count

    except Exception as e:
        logger.error(f"Failed to get PDF page count: {e}")
        return 0


def validate_pdf(pdf_source: Union[bytes, io.BytesIO, Path]) -> tuple[bool, str]:
    """
    PDFファイルの妥当性を検証

    Args:
        pdf_source: PDFのソース

    Returns:
        tuple[bool, str]: (検証成功, エラーメッセージ)
    """
    try:
        page_count = get_pdf_page_count(pdf_source)

        if page_count == 0:
            return False, "PDFにページが含まれていません"

        return True, ""

    except Exception as e:
        return False, f"PDF検証エラー: {str(e)}"
