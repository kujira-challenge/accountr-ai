#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF仕訳抽出システム - Streamlit Web App (State Machine Architecture)
100+ページ対応・ノンブロッキング・段階的処理
"""

# Initialize logging first
from logging_config import setup_logging
setup_logging()

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import time
import io
import tempfile
import shutil

# ローカルモジュール
from utils.processing_phases import ProcessingPhase, ProcessingState
from utils.split_phases import SplitPhase, SplitProcessingState
from utils.pdf_splitter import AdaptivePDFSplitter
from utils.pdf_utils import get_pdf_page_count, validate_pdf
from backend_processor_phase import PhaseBasedProcessor  # Phase3: 新しいプロセッサ
from backend_processor import convert_to_miroku_csv  # CSV変換用

# Import config safely with fallback
try:
    from config import config
except (ImportError, AttributeError) as config_error:
    st.error(f"⚠️ Configuration loading error: {config_error}")
    st.info("Please check your configuration files and restart the app.")
    st.stop()
import yaml

# ページ設定
st.set_page_config(
    page_title="PDF仕訳抽出システム",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== セッションステート初期化 =====
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = ProcessingState()

if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

if 'uploaded_pdf_bytes' not in st.session_state:
    st.session_state.uploaded_pdf_bytes = None

if 'phase_processor' not in st.session_state:
    st.session_state.phase_processor = None

if 'current_split_state' not in st.session_state:
    st.session_state.current_split_state = None

if 'llm_config' not in st.session_state:
    st.session_state.llm_config = {}

# ===== パスワード認証 =====
st.sidebar.markdown("---")
st.sidebar.markdown("🔐 **認証**")
password = st.sidebar.text_input("パスワードを入力してください", type="password")

try:
    app_password = st.secrets.get("APP_PASSWORD")
    if not app_password:
        st.sidebar.error("❌ APP_PASSWORDが設定されていません")
        st.error("🔐 システム管理者にお問い合わせください")
        st.info("💡 Streamlit Secrets で APP_PASSWORD を設定する必要があります")
        st.stop()

    if password != app_password:
        st.error("🚫 パスワードが正しくありません")
        st.info("💡 正しいパスワードを入力してアクセスしてください")
        st.stop()
    else:
        st.sidebar.success("✅ 認証成功")
except Exception as e:
    st.sidebar.error(f"❌ 認証システムエラー: {str(e)}")
    st.error("🔐 システム管理者にお問い合わせください")
    st.stop()

# ===== LLM設定 =====
st.sidebar.markdown("---")
st.sidebar.markdown("🤖 **LLM設定**")

@st.cache_data
def load_llm_config():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.sidebar.error(f"❌ 設定ファイル読込エラー: {e}")
        return {
            "llm": {"provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0.0},
            "pricing": {}
        }

cfg = load_llm_config()

# Provider selection (Geminiのみ)
providers = ["gemini"]
provider_index = 0
try:
    if cfg["llm"]["provider"] in providers:
        provider_index = providers.index(cfg["llm"]["provider"])
except (KeyError, ValueError):
    pass

provider = st.sidebar.selectbox(
    "LLMプロバイダ",
    providers,
    index=provider_index,
    help="Gemini APIを使用してPDFから仕訳データを抽出します"
)

# Model selection
models_by_provider = {
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"],
}

model_index = 0
try:
    current_models = models_by_provider[provider]
    if cfg["llm"]["model"] in current_models:
        model_index = current_models.index(cfg["llm"]["model"])
except (KeyError, ValueError):
    pass

model = st.sidebar.selectbox(
    "モデル",
    models_by_provider[provider],
    index=model_index,
    help="Flash系モデルはコストが安く、Pro系は精度重視"
)

# Temperature setting
temp = st.sidebar.slider(
    "Temperature",
    0.0, 1.0,
    value=float(cfg["llm"].get("temperature", 0.0)),
    step=0.1,
    help="0.0=決定的、1.0=創造的"
)

# Update session configuration
st.session_state.llm_config = {
    "provider": provider,
    "model": model,
    "temperature": temp
}

# サイドバー - システム情報
with st.sidebar:
    st.header("📊 システム情報")
    st.write(f"**AI Engine:** {provider.title()} ({model})")
    st.write(f"**分割単位:** Phase2最適化（3-5ページ）")
    st.write(f"**処理モード:** 🚀 ステップワイズ処理")
    st.caption("⏱️ Split単位タイムアウト: 120秒")

    # API設定確認
    try:
        if provider == "gemini":
            try:
                api_key = config.GOOGLE_API_KEY
            except AttributeError:
                import os
                api_key = os.environ.get("GOOGLE_API_KEY")

            if api_key:
                st.success("✅ Gemini API接続準備完了")
            else:
                st.error("❌ Gemini APIキーが未設定")
                st.warning("Settings > Secrets でGOOGLE_API_KEYを設定してください")
    except Exception as e:
        st.error(f"❌ API設定エラー: {str(e)}")
        st.info("💡 設定を確認してアプリを再起動してください")

    st.divider()
    st.caption(f"Powered by {provider.title()} {model}")

# ===== メイン処理フロー =====
st.title("📊 PDF仕訳抽出システム")
st.markdown("### 📄 PDFファイルから会計仕訳データを自動抽出してCSVで出力")

# 現在の処理状態
state = st.session_state.processing_state

# タイムアウトチェック（処理中のみ）
if state.phase in [ProcessingPhase.SPLITTING, ProcessingPhase.PROCESSING, ProcessingPhase.MERGING]:
    if state.is_timeout():
        logger.error(f"Timeout detected: {state.get_elapsed():.1f}s")
        state.phase = ProcessingPhase.TIMEOUT
        state.errors.append(f"処理がタイムアウトしました（{state.timeout_seconds}秒経過）")
        st.rerun()

# ===== フェーズ別処理 =====

# --- IDLE フェーズ: アップロード受付 ---
if state.phase == ProcessingPhase.IDLE:
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "📁 PDFファイルを選択してください",
            type=["pdf"],
            help="100ページ以上のPDFにも対応。自動的に最適なサイズで分割処理します。"
        )

    with col2:
        if uploaded_file:
            st.info(f"📄 **ファイル名:** {uploaded_file.name}")
            st.info(f"📊 **サイズ:** {uploaded_file.size / 1024 / 1024:.1f} MB")

            # ページ数取得
            page_count = get_pdf_page_count(uploaded_file)
            if page_count > 0:
                st.info(f"📖 **ページ数:** {page_count}ページ")

                # 処理時間の目安を表示
                estimated_time = page_count * 2  # 1ページあたり2秒の概算
                estimated_minutes = estimated_time // 60
                estimated_seconds = estimated_time % 60
                if estimated_minutes > 0:
                    st.caption(f"⏱️ 処理時間目安: 約{estimated_minutes}分{estimated_seconds}秒")
                else:
                    st.caption(f"⏱️ 処理時間目安: 約{estimated_seconds}秒")

    # APIキーチェック
    current_provider = st.session_state.llm_config.get("provider", "gemini")
    if current_provider == "gemini":
        try:
            google_api_key = config.GOOGLE_API_KEY
        except AttributeError:
            import os
            google_api_key = os.environ.get("GOOGLE_API_KEY")

        if not google_api_key:
            st.error("🚫 Gemini APIキーが設定されていません")
            st.info("📝 デプロイ後の設定が必要です。Streamlit SecretsでGOOGLE_API_KEYを設定してください。")
            st.stop()

    # 解析開始ボタン
    if uploaded_file is not None:
        st.divider()
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            current_model_display = f"{provider.title()} {model}"
            if st.button(
                "🚀 解析開始",
                use_container_width=True,
                type="primary",
                help=f"{current_model_display}で仕訳データの抽出を開始します"
            ):
                # PDF検証
                is_valid, error_msg = validate_pdf(uploaded_file)
                if not is_valid:
                    st.error(f"❌ PDF検証エラー: {error_msg}")
                else:
                    # 処理開始準備
                    uploaded_file.seek(0)
                    st.session_state.uploaded_pdf_bytes = uploaded_file.read()

                    # 一時ディレクトリ作成
                    st.session_state.temp_dir = tempfile.mkdtemp(prefix="pdf_splits_")

                    # 状態初期化
                    state.reset()
                    state.phase = ProcessingPhase.SPLITTING
                    state.pdf_name = uploaded_file.name
                    state.total_pages = get_pdf_page_count(io.BytesIO(st.session_state.uploaded_pdf_bytes))
                    state.start_time = time.time()

                    # config.yamlにLLM設定を保存
                    try:
                        with open("config.yaml", "w", encoding="utf-8") as f:
                            cfg_copy = cfg.copy()
                            cfg_copy["llm"]["provider"] = st.session_state.llm_config["provider"]
                            cfg_copy["llm"]["model"] = st.session_state.llm_config["model"]
                            cfg_copy["llm"]["temperature"] = st.session_state.llm_config["temperature"]
                            yaml.safe_dump(cfg_copy, f, default_flow_style=False, allow_unicode=True)
                        load_llm_config.clear()
                    except Exception as e:
                        logger.warning(f"Failed to update config.yaml: {e}")

                    logger.info(f"Processing started: {state.pdf_name}, {state.total_pages} pages")
                    st.rerun()

# --- SPLITTING フェーズ: PDF分割 ---
elif state.phase == ProcessingPhase.SPLITTING:
    st.info(f"📄 PDF分割中... ({state.total_pages}ページ)")

    try:
        # 一時ファイルに保存
        temp_pdf_path = Path(st.session_state.temp_dir) / state.pdf_name
        with open(temp_pdf_path, 'wb') as f:
            f.write(st.session_state.uploaded_pdf_bytes)

        # PDF分割実行
        splitter = AdaptivePDFSplitter(temp_dir=st.session_state.temp_dir)
        split_files, total_pages, pages_per_split = splitter.split_pdf(temp_pdf_path)

        # 状態更新
        state.split_files = [str(f) for f in split_files]
        state.total_splits = len(split_files)
        state.pages_per_split = pages_per_split
        state.current_split_index = 0
        state.split_results = []

        logger.info(f"PDF split completed: {state.total_splits} splits, {pages_per_split} pages/split")

        # 次フェーズへ
        state.phase = ProcessingPhase.PROCESSING
        st.rerun()

    except Exception as e:
        logger.error(f"PDF split failed: {e}", exc_info=True)
        state.phase = ProcessingPhase.ERROR
        state.errors.append(f"PDF分割エラー: {str(e)}")
        st.rerun()

# --- PROCESSING フェーズ: 分割単位で処理（Phase3: フェーズベース） ---
elif state.phase == ProcessingPhase.PROCESSING:
    # プログレスバー表示
    progress = state.get_progress_percentage()
    st.progress(progress, text=f"処理中... {state.current_split_index}/{state.total_splits} 分割完了")

    # 処理情報表示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("進捗", f"{state.current_split_index}/{state.total_splits}")
    with col2:
        st.metric("成功", state.get_successful_splits_count())
    with col3:
        st.metric("経過時間", state.get_elapsed_str())

    # タイムアウト警告（残り5分）
    elapsed = state.get_elapsed()
    remaining = state.timeout_seconds - elapsed
    if remaining <= 300 and remaining > 0:
        st.warning(f"⏰ 残り時間: 約{int(remaining/60)}分")

    # Phase3: フェーズ停滞チェック
    if state.is_phase_stalled():
        logger.error(f"Phase stalled: {state.phase_stall_count} consecutive stalls")
        state.phase = ProcessingPhase.ERROR
        state.errors.append("処理が停滞しました（同じフェーズで進捗なし）")
        st.rerun()

    # 1つの分割を処理（Phase3: フェーズごとに分解）
    if state.current_split_index < state.total_splits:
        # PhaseBasedProcessor初期化（遅延初期化）
        if st.session_state.phase_processor is None:
            st.session_state.phase_processor = PhaseBasedProcessor()

        processor = st.session_state.phase_processor
        split_path = Path(state.split_files[state.current_split_index])

        # 現在の分割の処理状態を取得または作成
        if st.session_state.current_split_state is None:
            # ページ範囲を抽出
            filename = split_path.stem
            page_range = "unknown"
            if '_pages_' in filename:
                try:
                    page_range = filename.split('_pages_')[1]
                except:
                    pass

            parts = page_range.split('-')
            if len(parts) == 2:
                try:
                    page_start = int(parts[0])
                    page_end = int(parts[1])
                except:
                    page_start = state.current_split_index * state.pages_per_split + 1
                    page_end = (state.current_split_index + 1) * state.pages_per_split
            else:
                page_start = state.current_split_index * state.pages_per_split + 1
                page_end = (state.current_split_index + 1) * state.pages_per_split

            # 新しい分割状態を作成
            st.session_state.current_split_state = SplitProcessingState(
                split_index=state.current_split_index,
                split_path=str(split_path),
                page_start=page_start,
                page_end=page_end,
                phase=SplitPhase.GEMINI_CALL
            )
            logger.info(f"Created new split state for split {state.current_split_index+1}/{state.total_splits}")

        split_state = st.session_state.current_split_state

        # 現在のフェーズを表示
        phase_display = {
            SplitPhase.GEMINI_CALL: "🤖 Gemini API 呼び出し中",
            SplitPhase.JSON_PARSE: "📊 JSON パース中",
            SplitPhase.POSTPROCESS: "🔧 データ後処理中",
            SplitPhase.VALIDATION: "✅ データ検証中",
            SplitPhase.COMPLETED: "✓ 完了",
            SplitPhase.FAILED: "❌ 失敗"
        }
        current_phase_text = phase_display.get(split_state.phase, split_state.phase.value)

        st.info(f"📄 分割 {state.current_split_index+1}/{state.total_splits}: {current_phase_text}")

        # Phase3: 1フェーズだけ処理
        with st.spinner(f"{current_phase_text}..."):
            try:
                # 1フェーズ処理
                result = processor.process_phase(
                    split_state=split_state,
                    split_path=split_path,
                    total_splits=state.total_splits
                )

                # 結果を処理
                if result["split_complete"]:
                    # 分割処理完了（成功 or 失敗）
                    logger.info(
                        f"Split {state.current_split_index+1}/{state.total_splits} complete: "
                        f"success={result['success']}"
                    )

                    # 結果を保存
                    final_data = split_state.get_final_data()
                    state.split_results.append({
                        "success": result["success"],
                        "data": final_data,
                        "error": result.get("error"),
                        "processing_time": 0.0,
                        "split_info": {
                            "index": state.current_split_index,
                            "filename": split_path.name,
                            "pages": f"{split_state.page_start}-{split_state.page_end}"
                        },
                        "entries_count": len(final_data),
                        "timeout": False
                    })

                    # エラーの場合は表示
                    if not result["success"]:
                        st.error(f"❌ 分割 {state.current_split_index+1} でエラーが発生しました")
                        with st.expander("🔍 エラー詳細", expanded=False):
                            st.code(result.get("error", "不明なエラー"))
                        st.caption("このページの処理をスキップして次へ進みます...")
                        state.errors.append(f"Split {state.current_split_index+1}: {result.get('error')}")
                        time.sleep(2)

                    # 次の分割へ
                    state.current_split_index += 1
                    st.session_state.current_split_state = None  # リセット
                    state.phase_stall_count = 0  # リセット
                    st.rerun()

                else:
                    # フェーズ完了、次のフェーズへ
                    logger.debug(f"Phase {split_state.phase.value} complete, continuing to next phase")
                    state.phase_stall_count = 0  # 進捗があったのでリセット
                    st.rerun()

            except Exception as e:
                # Phase3: 予期しないエラー
                logger.exception(f"Unexpected error in phase processing: {e}")

                st.error(f"❌ 予期しないエラーが発生しました")
                with st.expander("🔍 エラー詳細", expanded=True):
                    st.code(str(e))
                    st.caption(f"📄 ファイル: {split_path.name}")
                    st.caption(f"📍 フェーズ: {split_state.phase.value}")

                # エラー結果を保存
                state.split_results.append({
                    "success": False,
                    "data": [],
                    "error": f"予期しないエラー: {str(e)}",
                    "processing_time": 0.0,
                    "split_info": {
                        "index": state.current_split_index,
                        "filename": split_path.name,
                        "pages": f"{split_state.page_start}-{split_state.page_end}"
                    },
                    "entries_count": 0,
                    "timeout": False
                })

                state.errors.append(f"Unexpected error in split {state.current_split_index+1}: {str(e)}")

                # エラー継続か停止かをユーザーに選択させる
                col_err1, col_err2 = st.columns(2)
                with col_err1:
                    if st.button("⏭️ スキップして次へ", type="secondary"):
                        state.current_split_index += 1
                        st.session_state.current_split_state = None
                        st.rerun()
                with col_err2:
                    if st.button("🛑 処理を中止", type="primary"):
                        state.phase = ProcessingPhase.ERROR
                        st.rerun()

                # 処理を停止（ユーザーの選択を待つ）
                st.stop()

    else:
        # 全分割完了 → MERGING へ
        logger.info(f"All splits processed: {len(state.split_results)} results")
        state.phase = ProcessingPhase.MERGING
        st.rerun()

# --- MERGING フェーズ: 結果統合 ---
elif state.phase == ProcessingPhase.MERGING:
    st.info("📊 結果を統合中...")

    try:
        # Phase3: 直接split_resultsからデータを統合
        all_data = []
        successful_splits = 0
        failed_splits = 0

        for result in state.split_results:
            if result.get("success", False):
                successful_splits += 1
                if result.get("data"):
                    all_data.extend(result["data"])
            else:
                failed_splits += 1

        merged_result = {
            "success": successful_splits > 0,
            "all_data": all_data,
            "total_entries": len(all_data),
            "successful_splits": successful_splits,
            "failed_splits": failed_splits,
            "total_processing_time": state.get_elapsed()
        }

        if not merged_result["success"]:
            raise Exception("全ての分割処理が失敗しました")

        all_data = merged_result["all_data"]
        total_entries = merged_result["total_entries"]

        logger.info(f"Merge completed: {total_entries} entries")

        # CSV変換
        if total_entries > 0:
            df, csv_bytes, processing_info = convert_to_miroku_csv(all_data)

            # 最終結果を保存
            state.final_df = df
            state.final_csv_bytes = csv_bytes
            state.processing_info = processing_info
            state.processing_info['total_processing_time'] = merged_result['total_processing_time']
            state.processing_info['successful_splits'] = merged_result['successful_splits']
            state.processing_info['failed_splits'] = merged_result['failed_splits']
        else:
            # データなし
            state.final_df = pd.DataFrame()
            state.final_csv_bytes = b""
            state.processing_info = {
                'total_processing_time': merged_result['total_processing_time'],
                'successful_splits': merged_result['successful_splits'],
                'failed_splits': merged_result['failed_splits']
            }

        # 次フェーズへ
        state.phase = ProcessingPhase.COMPLETED
        st.rerun()

    except Exception as e:
        logger.error(f"Merge failed: {e}", exc_info=True)
        state.phase = ProcessingPhase.ERROR
        state.errors.append(f"結果統合エラー: {str(e)}")
        st.rerun()

# --- COMPLETED フェーズ: 結果表示 ---
elif state.phase == ProcessingPhase.COMPLETED:
    # クリーンアップ
    if st.session_state.temp_dir and Path(st.session_state.temp_dir).exists():
        try:
            shutil.rmtree(st.session_state.temp_dir)
            logger.info(f"Temp directory cleaned up: {st.session_state.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir: {e}")
        st.session_state.temp_dir = None

    # 結果表示
    df = state.final_df
    csv_bytes = state.final_csv_bytes
    processing_info = state.processing_info

    # 成功メッセージ
    total_time = state.get_elapsed()
    successful_splits = processing_info.get('successful_splits', 0)
    failed_splits = processing_info.get('failed_splits', 0)

    if failed_splits > 0:
        st.warning(f"⚠️ 処理完了（一部エラーあり）: {successful_splits}成功 / {failed_splits}失敗")
    else:
        st.success(f"🎉 抽出が完了しました！処理時間: {total_time:.1f}秒")

    # エラー詳細
    zero_errors = processing_info.get('zero_amount_errors', 0)
    missing_codes = processing_info.get('missing_codes_count', 0)

    if zero_errors > 0 or missing_codes > 0:
        with st.expander("⚠️ データ品質に関する注意事項", expanded=True):
            if zero_errors > 0:
                st.error(f"🚫 金額読取不可エラー: {zero_errors}件")
                st.caption("金額が0または読み取れなかった行はCSVから除外されました")

            if missing_codes > 0:
                st.warning(f"🔍 科目コード未割当: {missing_codes}件")
                st.caption("摘要に【科目コード要確認】が付記された行があります。手動で科目コードを設定してください")

    # 処理メトリクス
    metrics = processing_info.get('metrics', {})
    if metrics and any(v > 0 for v in metrics.values()):
        with st.expander("📊 処理統計・監査情報", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("🔄 前段整形")
                if metrics.get('one_vs_many_splits', 0) > 0:
                    st.metric("one-vs-many分割", metrics['one_vs_many_splits'])
                if metrics.get('left_right_swaps', 0) > 0:
                    st.metric("左右入替", metrics['left_right_swaps'])
                if metrics.get('sum_rows_dropped', 0) > 0:
                    st.metric("合算行除去", metrics['sum_rows_dropped'])

            with col2:
                st.subheader("🔚 後段整形")
                if metrics.get('empty_codes_excluded', 0) > 0:
                    st.metric("両コード空除外", metrics['empty_codes_excluded'])
                if metrics.get('duplicates_excluded', 0) > 0:
                    st.metric("重複圧縮", metrics['duplicates_excluded'])
                if metrics.get('unassigned_codes', 0) > 0:
                    st.metric("未割当要確認", metrics['unassigned_codes'])

            with col3:
                st.subheader("📈 処理サマリ")
                st.metric("総分割数", state.total_splits)
                st.metric("成功分割", successful_splits)
                st.metric("失敗分割", failed_splits)

    # 結果サマリー
    col_result1, col_result2, col_result3 = st.columns(3)
    with col_result1:
        st.metric("抽出エントリ数", len(df) if df is not None else 0)
    with col_result2:
        st.metric("処理時間", f"{total_time:.1f}秒")
    with col_result3:
        st.metric("処理ページ数", state.total_pages)

    # データプレビュー
    if df is not None and not df.empty:
        st.divider()
        st.subheader("📋 ミロク取込45列CSV プレビュー")
        st.info("🔄 抽出された5カラムJSON → ミロク取込45列CSV に変換済み（科目コード自動補完）")

        # 表示件数選択
        display_count = st.selectbox(
            "表示件数を選択",
            [10, 25, 50, 100, len(df)],
            index=1,
            key="display_count"
        )

        # データ表示（UI用にマスキング）
        from utils.masking import mask_personal_info
        display_df = df.head(display_count).copy()

        # 摘要列をマスキング
        if '摘要' in display_df.columns:
            display_df['摘要'] = display_df['摘要'].apply(lambda x: mask_personal_info(str(x)) if pd.notna(x) else x)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption("※ UI表示では個人識別情報をマスクしています。CSVファイルには元データが保存されます。")

        if len(df) > display_count:
            st.info(f"表示: {display_count}件 / 全{len(df)}件")
    else:
        st.warning("⚠️ 抽出結果が空でした。PDFの内容をご確認ください。")

    # ダウンロードボタン
    st.divider()
    if df is not None and len(df) > 0:
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            download_filename = f"{Path(state.pdf_name).stem}_mjs45_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button(
                label="📥 ミロク取込45列CSV をダウンロード",
                data=csv_bytes,
                file_name=download_filename,
                mime="text/csv",
                use_container_width=True,
                type="secondary",
                help="ミロク会計システムに直接取り込み可能な45列形式のCSVファイル"
            )
    else:
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.button(
                "📥 45列CSVをダウンロード (データなし)",
                disabled=True,
                use_container_width=True,
                help="抽出されたデータがありません"
            )

    # リセットボタン
    st.divider()
    col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
    with col_reset2:
        if st.button("🔄 新しいPDFを処理", use_container_width=True):
            state.reset()
            st.session_state.uploaded_pdf_bytes = None
            st.session_state.phase_processor = None
            st.session_state.current_split_state = None
            st.rerun()

# --- ERROR フェーズ: エラー表示 ---
elif state.phase == ProcessingPhase.ERROR:
    # クリーンアップ
    if st.session_state.temp_dir and Path(st.session_state.temp_dir).exists():
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir: {e}")
        st.session_state.temp_dir = None

    st.error("❌ 処理中にエラーが発生しました")

    # エラー詳細
    with st.expander("🔍 エラー詳細", expanded=True):
        for i, err in enumerate(state.errors, 1):
            st.text(f"{i}. {err}")

    # リセットボタン
    col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
    with col_reset2:
        if st.button("🔄 最初から やり直す", use_container_width=True, type="primary"):
            state.reset()
            st.session_state.uploaded_pdf_bytes = None
            st.session_state.phase_processor = None
            st.session_state.current_split_state = None
            st.rerun()

# --- TIMEOUT フェーズ: タイムアウト表示 ---
elif state.phase == ProcessingPhase.TIMEOUT:
    # クリーンアップ
    if st.session_state.temp_dir and Path(st.session_state.temp_dir).exists():
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir: {e}")
        st.session_state.temp_dir = None

    st.error(f"⏰ 処理がタイムアウトしました（制限時間: {state.timeout_seconds//60}分）")

    # 進捗情報
    st.info(f"📊 処理進捗: {state.current_split_index}/{state.total_splits} 分割完了")

    # 部分結果があれば表示
    if state.split_results:
        st.warning("⚠️ 処理途中までのデータがあります。部分的に処理できた可能性があります。")

        successful = state.get_successful_splits_count()
        failed = state.get_failed_splits_count()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("成功した分割", successful)
        with col2:
            st.metric("失敗した分割", failed)

    # リセットボタン
    st.divider()
    col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
    with col_reset2:
        if st.button("🔄 最初から やり直す", use_container_width=True, type="primary"):
            state.reset()
            st.session_state.uploaded_pdf_bytes = None
            st.session_state.phase_processor = None
            st.session_state.current_split_state = None
            st.rerun()

# ===== 使用方法とヒント =====
st.divider()
with st.expander("📖 使用方法とヒント"):
    st.markdown("""
    ### 📋 基本的な使い方
    1. **PDFファイルをアップロード**
       - 上の「📁 PDFファイルを選択してください」エリアをクリック
       - または、ファイルをドラッグ&ドロップ
       - 対応形式: PDFファイル (.pdf)
       - **100ページ以上のPDFにも対応**

    2. **解析開始**
       - 「🚀 解析開始」ボタンをクリック
       - 処理は段階的に進み、進捗状況が表示されます
       - 画面を閉じずにお待ちください

    3. **結果を確認**
       - 抽出された仕訳データが表示されます
       - 摘要欄の内容を確認し、必要に応じて後で修正してください

    4. **CSVファイルをダウンロード**
       - 「📥 ミロク取込45列CSVをダウンロード」ボタンをクリック
       - ダウンロードしたCSVファイルをミロク会計システムに取り込み

    ### 🚀 ステップワイズ処理の特徴（Phase2最適化版）
    - **大規模PDF対応**: 100ページ以上のPDFも確実に完走
    - **適応型分割**: PDFサイズに応じて最適な分割サイズを自動決定
      - 小規模（~30ページ）: 5ページずつ
      - 中規模（~100ページ）: 4ページずつ
      - 大規模（100ページ~）: 3ページずつ
    - **UI非ブロッキング**: 処理中も進捗状況をリアルタイム表示
    - **2段階タイムアウト保護**:
      - Split単位: 120秒（1分割あたり）
      - 全体: 15分（残り5分で警告表示）
    - **エラー耐性**: 一部の分割が失敗しても処理続行
    - **エラー可視化**: エラー発生時は必ずUIに詳細を表示

    ### 📊 このシステムの会計処理ロジック
    **複式簿記の原則:**
    - ✓ 1取引 = 借方1本以上 + 貸方1本以上
    - ✓ 借方合計 = 貸方合計（必ず一致）
    - ✓ 借方のみ・貸方のみの単一仕訳は禁止

    **出力項目:**
    1. 伝票日付（取引日）
    2. 借貸区分（「借方」または「貸方」）
    3. 科目名（勘定科目）
    4. 金額（正の整数、カンマ除去済み）
    5. 摘要（取引内容、契約者名、物件名など）

    **科目コードの割当:**
    - 抽出された科目名を「勘定科目コード一覧.csv」と照合し、自動的に科目コードを割り当て
    - 照合方法: ①完全一致 → ②エイリアス（揺らぎ対応） → ③部分一致
    - 未割当の場合: 摘要欄に「【科目コード要確認】」を付記

    ### ⚠️ アップロード時の注意点
    - **推奨ファイル**: 明細が表形式で記載された通帳・請求書・領収書のPDF
    - **非推奨**: 手書き文字、スキャン品質が低い、文字が不鮮明なPDF
    - **処理時間**: ページ数に応じて時間がかかります（目安: 1ページあたり2秒）
    - **タイムアウト**: 15分以内に処理が完了しない場合は自動停止

    ### 🔧 トラブルシューティング
    **Q: 処理がタイムアウトする**
    - 非常に大きなPDF（200ページ以上）の場合、タイムアウトする可能性があります
    - PDFを分割して複数回に分けて処理してください

    **Q: 一部の分割が失敗する**
    - 画質が悪いページがある場合、そのページの処理が失敗することがあります
    - 成功した分割のデータは正常に抽出されます

    **Q: エラーが表示される**
    - 「🔄 最初からやり直す」ボタンで処理をリセットできます
    - それでも解決しない場合は、PDFファイルを確認してください

    ### 🔒 セキュリティとプライバシー
    - アップロードされたPDFは一時的に処理され、完了後に自動削除されます
    - 処理完了後、サーバーにはデータが残りません
    - 個人情報を含むデータは慎重に取り扱ってください
    """)

# フッター
st.divider()
st.caption("📊 PDF仕訳抽出システム | Powered by Gemini | Built with Streamlit")
