#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Backend Processor
PDFをアップロードしてCSV形式で仕訳データを出力する
"""

import io
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import logging

# ローカルモジュール
from pdf_extractor import ProductionPDFExtractor
from config import config

# ログ設定
logger = logging.getLogger(__name__)

def process_pdf_to_csv(uploaded_file) -> Tuple[pd.DataFrame, bytes, Dict[str, Any]]:
    """
    アップロードされたPDFファイルを処理してCSVデータを返す
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple[pd.DataFrame, bytes, Dict[str, Any]]: (データフレーム, CSV bytes, 処理情報)
    """
    logger.info(f"Processing uploaded PDF: {uploaded_file.name}")
    
    # 一時ディレクトリの作成
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # アップロードファイルを一時保存
        pdf_path = temp_path / uploaded_file.name
        with open(pdf_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        logger.info(f"PDF saved temporarily to: {pdf_path}")
        
        # PDF抽出器を初期化（Claude Sonnet 4.0使用）
        extractor = ProductionPDFExtractor(
            api_key=config.ANTHROPIC_API_KEY,
            use_mock=config.USE_MOCK_DATA
        )
        
        # PDF処理実行
        result = extractor.process_single_pdf(
            pdf_path, 
            temp_path / "processing_temp",
            pages_per_split=config.PAGES_PER_SPLIT
        )
        
        if not result.success:
            raise Exception(f"PDF processing failed: {result.error_message}")
        
        logger.info(f"Successfully processed PDF: {len(result.data)} entries extracted")
        
        # DataFrameに変換
        if result.data:
            df = pd.DataFrame(result.data)
            
            # CSV列の順序を整える
            if hasattr(config, 'CSV_COLUMNS'):
                # 設定された列順序を使用
                missing_cols = [col for col in config.CSV_COLUMNS if col not in df.columns]
                for col in missing_cols:
                    df[col] = ""
                df = df[config.CSV_COLUMNS]
            
        else:
            # 空のデータフレームを作成
            columns = getattr(config, 'CSV_COLUMNS', [
                "契約日", "借方科目", "貸方科目", "摘要", "金額", "備考", "参照元ファイル", "ページ"
            ])
            df = pd.DataFrame(columns=columns)
        
        # CSV bytes形式で出力
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
        
        logger.info(f"CSV data prepared: {len(df)} rows, {len(csv_bytes)} bytes")
        
        # 処理情報を準備
        processing_info = {
            "cost_usd": result.total_cost_usd,
            "cost_jpy": result.total_cost_jpy,
            "processing_time": result.processing_time,
            "pages_processed": result.pages_processed,
            "entries_extracted": len(df)
        }
        
        return df, csv_bytes, processing_info