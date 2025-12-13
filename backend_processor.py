#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Backend Processor
PDFをアップロードしてCSV形式で仕訳データを出力する
"""

# Initialize logging first
from logging_config import setup_logging
setup_logging()

import io
import json
import tempfile
import os
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import logging
from datetime import datetime

# ローカルモジュール
from pdf_extractor import ProductionPDFExtractor
from mjs_converter import fivejson_to_mjs45, MJSConverter
from config import config

# ログ設定
logger = logging.getLogger(__name__)

def create_diag_snapshot(raw_response, parsed_data, reconciled_data, mjs45_csv, finalized_csv, error_msg):
    """
    0件事故時のスナップショット作成
    
    Args:
        raw_response: LLMの生レスポンス
        parsed_data: 防御パース後のJSON
        reconciled_data: 前段整形後のJSON
        mjs45_csv: MJS45中間CSV
        finalized_csv: 後段整形直前CSV
        error_msg: エラーメッセージ
        
    Returns:
        str: スナップショットディレクトリのパス
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_dir = Path(f"output/tmp/diag_{timestamp}")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # LLMの生レスポンス保存
        if raw_response:
            with open(snapshot_dir / "raw_response.txt", 'w', encoding='utf-8') as f:
                f.write(raw_response)
        
        # 各段階のデータ保存
        stages = [
            ("parsed_data.json", parsed_data),
            ("reconciled_data.json", reconciled_data),
            ("mjs45_intermediate.csv", mjs45_csv),
            ("finalized_csv_pre.csv", finalized_csv)
        ]
        
        for filename, data in stages:
            if data is not None:
                if filename.endswith('.json'):
                    with open(snapshot_dir / filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                elif filename.endswith('.csv') and hasattr(data, 'to_csv'):
                    data.to_csv(snapshot_dir / filename, index=False, encoding='utf-8-sig')
        
        # エラー情報保存
        with open(snapshot_dir / "error_info.txt", 'w', encoding='utf-8') as f:
            f.write(f"Error: {error_msg}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        
        logger.info(f"Diagnostic snapshot created: {snapshot_dir}")
        return str(snapshot_dir)
    except Exception as e:
        logger.error(f"Failed to create diagnostic snapshot: {e}")
        return None

def convert_to_miroku_csv(json_data: list) -> Tuple[pd.DataFrame, bytes, Dict[str, Any]]:
    """
    5カラムJSONデータをMJS45列CSV形式に変換

    Args:
        json_data: 5カラム形式のJSONデータリスト

    Returns:
        Tuple[pd.DataFrame, bytes, Dict[str, Any]]: (データフレーム, CSV bytes, 処理情報)
    """
    logger.info(f"Converting {len(json_data)} entries to MJS45 CSV format")

    try:
        # 一時ディレクトリの作成
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # ========== DIAG: stage1_parse ==========
            stage1_count = len(json_data) if json_data else 0
            logger.info(f"DIAG stage1_parse: count={stage1_count}")
            if stage1_count == 0:
                raise RuntimeError("DIAG stage1_parse=0: 入力JSONデータが空です。")

            # 5カラムJSON→45列MJS CSV変換
            if json_data:
                # 1. 5カラムJSONを一時保存
                json_path = temp_path / "extracted_data.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

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
                    mjs_csv_path = temp_path / "mjs45_output.csv"
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

                            # ========== DIAG: stage4_mjs45 ==========
                            stage4_count = len(df)
                            logger.info(f"DIAG stage4_mjs45: count={stage4_count}")
                            if stage4_count == 0:
                                raise RuntimeError("DIAG stage4_mjs45=0: コード割当結果が空。名寄せ/マスタ読込を確認してください。")

                            logger.info(f"MJS 45-column CSV loaded: {stage4_count} rows")
                        else:
                            logger.error("MJS CSV file was not created")
                            df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)

                            # ========== DIAG: stage4_mjs45 ==========
                            logger.error(f"DIAG stage4_mjs45: count=0")
                            raise RuntimeError("DIAG stage4_mjs45=0: MJSファイルが作成されませんでした。")

                    except Exception as e:
                        logger.error(f"MJS conversion failed: {e}")
                        # フォールバック: 空の45列DataFrame
                        df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)

            else:
                # ========== DIAG: stage1_parse ==========
                logger.error(f"DIAG stage1_parse: count=0")
                raise RuntimeError("DIAG stage1_parse=0: 入力データが空。")

                # 空の45列DataFrame作成
                df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)

            # 最終整形処理（CSV直前） - 両コード空除去とCSV重複削除のみ
            if not df.empty:
                try:
                    from utils.finalize_csv import finalize_csv_rows

                    # DataFrameを辞書リストに変換
                    csv_rows = df.to_dict('records')
                    original_count = len(csv_rows)
                    logger.info(f"Starting final CSV cleanup for {original_count} rows")

                    # 両コード空除去と重複除去
                    finalized_rows = finalize_csv_rows(csv_rows)

                    # ========== DIAG: stage5_finalize ==========
                    stage5_count = len(finalized_rows)
                    logger.info(f"DIAG stage5_finalize: count={stage5_count}")
                    if stage5_count == 0:
                        # スナップショット作成
                        snapshot_dir = create_diag_snapshot(
                            raw_response=None,
                            parsed_data=json_data,
                            reconciled_data=None,
                            mjs45_csv=df,
                            finalized_csv=pd.DataFrame(csv_rows),
                            error_msg="stage5_finalize=0: 後段整形で全除去"
                        )
                        raise RuntimeError(f"DIAG stage5_finalize=0: 後段整形で全除去。条件過剰 or 前段/名寄せ不整合。snapshot={snapshot_dir}")

                    # 整形後のDataFrameに変換
                    if finalized_rows:
                        df = pd.DataFrame(finalized_rows)
                        # 必要な列が不足している場合は補完
                        for col in MJSConverter.MJS_45_COLUMNS:
                            if col not in df.columns:
                                df[col] = ""
                        # 列順序を45列形式に合わせる
                        df = df.reindex(columns=MJSConverter.MJS_45_COLUMNS, fill_value="")

                        # 仕訳No（伝票NO）順にソート
                        if "伝票NO" in df.columns and not df.empty:
                            logger.info("Sorting by 伝票NO (voucher number)")
                            # ソートキーの作成：伝票NO → 借方・貸方の順（借方優先）
                            # 借方・貸方の判定：（借）科目コードがある = 借方, （貸）科目コードがある = 貸方
                            df['_sort_debit_credit'] = df.apply(
                                lambda row: 0 if row.get('（借）科目ｺｰﾄﾞ', '') != '' else 1,
                                axis=1
                            )
                            # 伝票NOを数値としてソート（空文字は最後）
                            df['_sort_voucher_no'] = pd.to_numeric(df['伝票NO'], errors='coerce').fillna(float('inf'))

                            df = df.sort_values(
                                by=['_sort_voucher_no', '_sort_debit_credit'],
                                ascending=[True, True]
                            )

                            # ソート用の一時列を削除
                            df = df.drop(columns=['_sort_debit_credit', '_sort_voucher_no'])
                            logger.info(f"Sorted {len(df)} rows by voucher number")

                        dropped_count = original_count - len(finalized_rows)
                        if dropped_count > 0:
                            logger.info(f"Final CSV cleanup: {original_count} -> {len(finalized_rows)} rows (dropped {dropped_count} rows with empty codes)")
                        else:
                            logger.info(f"Final CSV cleanup completed: {len(finalized_rows)} rows")
                    else:
                        logger.warning("Final CSV cleanup returned no rows")
                        df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)

                except Exception as e:
                    logger.error(f"Final CSV cleanup failed: {e}")
                    # エラー時は元のDataFrameをそのまま使用
                    pass

            # CSV bytes形式で出力
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')

            logger.info(f"CSV data prepared: {len(df)} rows, {len(csv_bytes)} bytes")

            # 科目コード未割当の件数を取得
            missing_codes_count = len([row for _, row in df.iterrows() if '【科目コード要確認】' in str(row.get('摘要', ''))])

            # メトリクス情報を収集
            metrics_info = {
                # ステージ別件数
                "stage1_count": stage1_count if 'stage1_count' in locals() else 0,
                "stage4_count": len(df) if 'stage4_count' in locals() else 0,
                "stage5_count": len(df),

                # 後段整形メトリクス
                "unassigned_codes": missing_codes_count
            }

            # 処理情報を準備
            processing_info = {
                "entries_extracted": len(df),
                "missing_codes_count": missing_codes_count,
                "zero_amount_errors": 0,  # ステップワイズ処理では別途管理
                "metrics": metrics_info
            }

            return df, csv_bytes, processing_info

    except Exception as e:
        logger.error(f"CSV conversion error: {e}", exc_info=True)
        # エラー時も3つの戻り値を返す - 45列空DataFrame
        empty_df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)
        csv_buffer = io.StringIO()
        empty_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        empty_csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')

        error_info = {
            "entries_extracted": 0,
            "error": str(e),
            "metrics": {}
        }

        return empty_df, empty_csv_bytes, error_info


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
            
            # ========== DIAG: stage1_parse ==========
            stage1_count = len(result.data) if result.data else 0
            logger.info(f"DIAG stage1_parse: count={stage1_count}")
            if stage1_count == 0:
                raise RuntimeError("DIAG stage1_parse=0: 防御パース後の5カラム配列が空。PDF読取またはLLM抽出に失敗。")
            
            # サンプル摘要をログに表示
            sample_memos = [entry.get("摘要", "")[:50] for entry in result.data[:2]]
            logger.info(f"DIAG stage1_parse samples: {sample_memos}")
            
            logger.info(f"Successfully processed PDF: {stage1_count} entries extracted")
            
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
                            
                            # ========== DIAG: stage4_mjs45 ==========
                            stage4_count = len(df)
                            logger.info(f"DIAG stage4_mjs45: count={stage4_count}")
                            if stage4_count == 0:
                                raise RuntimeError("DIAG stage4_mjs45=0: コード割当結果が空。名寄せ/マスタ読込を確認してください。")
                            
                            logger.info(f"MJS 45-column CSV loaded: {stage4_count} rows")
                        else:
                            logger.error("MJS CSV file was not created")
                            df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)
                            
                            # ========== DIAG: stage4_mjs45 ==========
                            logger.error(f"DIAG stage4_mjs45: count=0")
                            raise RuntimeError("DIAG stage4_mjs45=0: MJSファイルが作成されませんでした。")
                        
                    except Exception as e:
                        logger.error(f"MJS conversion failed: {e}")
                        # フォールバック: 空の45列DataFrame
                        df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)
                
            else:
                # ========== DIAG: stage1_parse ==========
                logger.error(f"DIAG stage1_parse: count=0")
                raise RuntimeError("DIAG stage1_parse=0: PDF抽出結果が空。")
                
                # 空の45列DataFrame作成
                df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)
            
            # 最終整形処理（CSV直前） - 両コード空除去とCSV重複削除のみ
            if not df.empty:
                try:
                    from utils.finalize_csv import finalize_csv_rows
                    
                    # DataFrameを辞書リストに変換
                    csv_rows = df.to_dict('records')
                    original_count = len(csv_rows)
                    logger.info(f"Starting final CSV cleanup for {original_count} rows")
                    
                    # 両コード空除去と重複除去
                    finalized_rows = finalize_csv_rows(csv_rows)
                    
                    # ========== DIAG: stage5_finalize ==========
                    stage5_count = len(finalized_rows)
                    logger.info(f"DIAG stage5_finalize: count={stage5_count}")
                    if stage5_count == 0:
                        # スナップショット作成
                        snapshot_dir = create_diag_snapshot(
                            raw_response=None,  # PDF処理では生レスポンス取得困難
                            parsed_data=result.data if result else None,
                            reconciled_data=None,  # 前段整形データ取得困難
                            mjs45_csv=df,
                            finalized_csv=pd.DataFrame(csv_rows),
                            error_msg="stage5_finalize=0: 後段整形で全除去"
                        )
                        raise RuntimeError(f"DIAG stage5_finalize=0: 後段整形で全除去。条件過剰 or 前段/名寄せ不整合。snapshot={snapshot_dir}")
                    
                    # 整形後のDataFrameに変換
                    if finalized_rows:
                        df = pd.DataFrame(finalized_rows)
                        # 必要な列が不足している場合は補完
                        for col in MJSConverter.MJS_45_COLUMNS:
                            if col not in df.columns:
                                df[col] = ""
                        # 列順序を45列形式に合わせる
                        df = df.reindex(columns=MJSConverter.MJS_45_COLUMNS, fill_value="")

                        # 仕訳No（伝票NO）順にソート
                        if "伝票NO" in df.columns and not df.empty:
                            logger.info("Sorting by 伝票NO (voucher number)")
                            # ソートキーの作成：伝票NO → 借方・貸方の順（借方優先）
                            # 借方・貸方の判定：（借）科目コードがある = 借方, （貸）科目コードがある = 貸方
                            df['_sort_debit_credit'] = df.apply(
                                lambda row: 0 if row.get('（借）科目ｺｰﾄﾞ', '') != '' else 1,
                                axis=1
                            )
                            # 伝票NOを数値としてソート（空文字は最後）
                            df['_sort_voucher_no'] = pd.to_numeric(df['伝票NO'], errors='coerce').fillna(float('inf'))

                            df = df.sort_values(
                                by=['_sort_voucher_no', '_sort_debit_credit'],
                                ascending=[True, True]
                            )

                            # ソート用の一時列を削除
                            df = df.drop(columns=['_sort_debit_credit', '_sort_voucher_no'])
                            logger.info(f"Sorted {len(df)} rows by voucher number")
                        
                        dropped_count = original_count - len(finalized_rows)
                        if dropped_count > 0:
                            logger.info(f"Final CSV cleanup: {original_count} -> {len(finalized_rows)} rows (dropped {dropped_count} rows with empty codes)")
                        else:
                            logger.info(f"Final CSV cleanup completed: {len(finalized_rows)} rows")
                    else:
                        logger.warning("Final CSV cleanup returned no rows")
                        df = pd.DataFrame(columns=MJSConverter.MJS_45_COLUMNS)
                        
                except Exception as e:
                    logger.error(f"Final CSV cleanup failed: {e}")
                    # エラー時は元のDataFrameをそのまま使用
                    pass
            
            # CSV bytes形式で出力
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')
            
            logger.info(f"CSV data prepared: {len(df)} rows, {len(csv_bytes)} bytes")
            
            # エラー情報の収集
            error_entries = getattr(extractor, 'error_entries', [])
            missing_codes_count = len([row for _, row in df.iterrows() if '【科目コード要確認】' in str(row.get('摘要', ''))])
            
            # メトリクス情報を収集  
            metrics_info = {
                # ステージ別件数
                "stage1_count": stage1_count if 'stage1_count' in locals() else 0,
                "stage2_count": getattr(extractor, 'stage2_count', 0),
                "stage3_count": getattr(extractor, 'stage3_count', 0),
                "stage4_count": getattr(extractor, 'stage4_count', 0),
                "stage5_count": len(df),
                
                # 前段整形メトリクス
                "one_vs_many_splits": getattr(extractor, 'one_vs_many_splits', 0),
                "left_right_swaps": getattr(extractor, 'left_right_swaps', 0),
                "sum_rows_dropped": getattr(extractor, 'sum_rows_dropped', 0),
                
                # 後段整形メトリクス
                "empty_codes_excluded": getattr(extractor, 'empty_codes_excluded', 0),
                "duplicates_excluded": getattr(extractor, 'duplicates_excluded', 0),
                "unassigned_codes": missing_codes_count
            }
            
            # 処理情報を準備
            processing_info = {
                "cost_usd": getattr(result, 'total_cost_usd', 0.0),
                "cost_jpy": getattr(result, 'total_cost_jpy', 0.0),
                "processing_time": getattr(result, 'processing_time', 0.0),
                "pages_processed": getattr(result, 'pages_processed', 0),
                "entries_extracted": len(df),
                "error_entries": error_entries,
                "missing_codes_count": missing_codes_count,
                "zero_amount_errors": len(error_entries),
                "metrics": metrics_info
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