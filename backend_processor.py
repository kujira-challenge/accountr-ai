#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Backend Processor
PDFをアップロードしてCSV形式で仕訳データを出力する
"""

import io
import json
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import logging

# ローカルモジュール
from pdf_extractor import ProductionPDFExtractor
from mjs_converter import fivejson_to_mjs45, MJSConverter
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
    
    try:
        # 一時ディレクトリの作成
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # アップロードファイルを一時保存
            pdf_path = temp_path / uploaded_file.name
            with open(pdf_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            logger.info(f"PDF saved temporarily to: {pdf_path}")
            
            # PDF抽出器を初期化（Claude Sonnet 4.0使用、常に本番API）
            extractor = ProductionPDFExtractor(
                api_key=config.ANTHROPIC_API_KEY,
                use_mock=False  # モックモード完全撤廃
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
            
            # 5カラムJSON→45列MJS CSV変換
            if result.data:
                # 1. 5カラムJSONを一時保存
                json_path = temp_path / f"{uploaded_file.name}_extracted.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result.data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"5-column JSON saved: {json_path}")
                
                # 2. 勘定科目CSVファイルの存在確認
                account_code_csv_path = str(config.ACCOUNT_CODE_CSV_PATH)
                if not Path(account_code_csv_path).exists():
                    logger.error(f"Account code CSV file not found: {account_code_csv_path}")
                    # 空の45列DataFrameを作成
                    df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)
                    logger.warning("Using empty DataFrame due to missing account code CSV")
                else:
                    # 3. 45列CSVに変換
                    mjs_csv_path = temp_path / f"{uploaded_file.name}_mjs45.csv"
                    conversion_log_path = temp_path / "mjs_conversion.log"
                    
                    try:
                        fivejson_to_mjs45(
                            str(json_path),
                            account_code_csv_path,
                            str(mjs_csv_path),
                            str(conversion_log_path)
                        )
                        
                        # 4. 45列CSVを読み込んでDataFrameに変換
                        if mjs_csv_path.exists():
                            df = pd.read_csv(mjs_csv_path, encoding='utf-8-sig')
                            logger.info(f"MJS 45-column CSV loaded: {len(df)} rows")
                        else:
                            logger.error("MJS CSV file was not created")
                            df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)
                        
                    except Exception as e:
                        logger.error(f"MJS conversion failed: {e}")
                        # フォールバック: 空の45列DataFrame
                        df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)
                
            else:
                # 空の45列DataFrame作成
                df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)
            
            # CSV bytes形式で出力
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
            
            logger.info(f"CSV data prepared: {len(df)} rows, {len(csv_bytes)} bytes")
            
            # 処理情報を準備
            processing_info = {
                "cost_usd": getattr(result, 'total_cost_usd', 0.0),
                "cost_jpy": getattr(result, 'total_cost_jpy', 0.0),
                "processing_time": getattr(result, 'processing_time', 0.0),
                "pages_processed": getattr(result, 'pages_processed', 0),
                "entries_extracted": len(df)
            }
            
            return df, csv_bytes, processing_info
            
    except Exception as e:
        logger.error(f"PDF processing error: {e}", exc_info=True)
        # エラー時も3つの戻り値を返す - 45列空DataFrame
        empty_df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)
        csv_buffer = io.StringIO()
        empty_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        empty_csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
        
        error_info = {
            "cost_usd": 0.0,
            "cost_jpy": 0.0,
            "processing_time": 0.0,
            "pages_processed": 0,
            "entries_extracted": 0,
            "error": str(e)
        }
        
        # エラー情報を含む3つの戻り値を返す（エラーを再発生させない）
        return empty_df, empty_csv_bytes, error_info