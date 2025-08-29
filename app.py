#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF仕訳抽出システム - Streamlit Web App
Claude Sonnet 4.0を使用してPDFから会計仕訳データを抽出し、CSV形式で出力
"""

import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# ローカルモジュール
from backend_processor import process_pdf_to_csv
from config import config
import hashlib

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

# 🔐 パスワード認証チェック
st.sidebar.markdown("---")
st.sidebar.markdown("🔐 **認証**")
password = st.sidebar.text_input("パスワードを入力してください", type="password")

# パスワードチェック
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

# サイドバー - システム情報
with st.sidebar:
    st.header("📊 システム情報")
    st.write("**AI Engine:** Claude Sonnet 4.0")
    st.write(f"**分割単位:** {config.PAGES_PER_SPLIT}ページ")
    st.write(f"**処理モード:** {'🧪 テスト' if config.USE_MOCK_DATA else '🚀 本番'}")
    
    # API設定確認とエラーハンドリング
    try:
        api_key = config.ANTHROPIC_API_KEY
        if api_key and api_key != 'DUMMY_API_KEY':
            st.success("✅ Claude API接続準備完了")
        else:
            st.error("❌ Claude APIキーが未設定")
            st.warning("Streamlit Cloudの場合：Settings > Secrets でANTHROPIC_API_KEYを設定してください")
    except Exception as e:
        st.error(f"❌ API設定エラー: {str(e)}")
        st.info("💡 設定を確認してアプリを再起動してください")
    
    st.divider()
    st.caption("Powered by Claude Sonnet 4.0")

# セッションステート初期化（二重起動防止用）
if 'processing_cache' not in st.session_state:
    st.session_state.processing_cache = {}
if 'last_processed_hash' not in st.session_state:
    st.session_state.last_processed_hash = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# メインコンテンツ
st.title("📊 PDF仕訳抽出システム")
st.markdown("### 📄 PDFファイルから会計仕訳データを自動抽出してCSVで出力")

# アップロード部分
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📁 PDFファイルを選択してください",
        type=["pdf"],
        help="ページごとに分割処理されます（画像1枚=1リクエスト）。大きなファイルは時間がかかる場合があります。"
    )

with col2:
    if uploaded_file:
        st.info(f"📄 **ファイル名:** {uploaded_file.name}")
        st.info(f"📊 **サイズ:** {uploaded_file.size / 1024 / 1024:.1f} MB")

# APIキーチェック
if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == 'DUMMY_API_KEY':
    st.error("🚫 Claude APIキーが設定されていません")
    st.info("📝 デプロイ後の設定が必要です。README.mdの手順に従ってAPIキーを設定してください。")
    st.stop()

# 変換処理
if uploaded_file is not None:
    
    # ファイルハッシュ計算（二重起動防止用）
    file_content = uploaded_file.read()
    file_hash = hashlib.md5(file_content).hexdigest()
    uploaded_file.seek(0)  # ファイルポインタリセット
    
    # キャッシュチェック
    is_cached = file_hash in st.session_state.processing_cache
    cached_result = st.session_state.processing_cache.get(file_hash)
    
    # 概算費用計算（ファイルサイズ×係数）
    estimated_pages = max(1, int(uploaded_file.size / (1024 * 300)))  # 300KB/ページ仮定
    estimate_cost_usd = estimated_pages * 0.01  # 1ページあたり$0.01概算
    estimate_cost_jpy = estimate_cost_usd * config.get_current_usd_to_jpy_rate()
    
    # 概算コスト表示
    st.info(f"📊 **概算**: {estimated_pages}ページ予想 / 概算費用: ¥{estimate_cost_jpy:.0f} (${estimate_cost_usd:.3f} USD)")
    
    # 変換ボタン
    st.divider()
    col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
    
    with col_button2:
        # 既にキャッシュがある場合はボタンを無効化
        button_disabled = is_cached and cached_result is not None
        button_text = "✅ 抽出済み（結果を表示中）" if button_disabled else "🚀 仕訳データ抽出開始"
        
        convert_clicked = st.button(
            button_text,
            type="primary" if not button_disabled else "secondary",
            use_container_width=True,
            disabled=button_disabled,
            help="Claude Sonnet 4.0を使用してPDFから仕訳データを抽出します" if not button_disabled else "このファイルは既に処理済みです"
        )
    
    # 処理実行（キャッシュがある場合は結果を直接表示）
    if convert_clicked or (is_cached and cached_result):
        if is_cached and cached_result:
            # キャッシュから結果を取得
            df, csv_bytes, processing_info = cached_result
            st.info("💾 キャッシュから結果を表示しています（二重処理防止）")
        else:
            # 新規処理
            with st.spinner("🔄 Claude Sonnet 4.0で仕訳データを抽出中..."):
                try:
                    start_time = datetime.now()
                    
                    # プログレスバー表示
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📄 PDFファイルを分析中...")
                    progress_bar.progress(25)
                    
                    # メイン処理
                    df, csv_bytes, processing_info = process_pdf_to_csv(uploaded_file)
                    
                    # 結果をキャッシュに保存
                    st.session_state.processing_cache[file_hash] = (df, csv_bytes, processing_info)
                    st.session_state.last_processed_hash = file_hash
                    st.session_state.last_result = (df, csv_bytes, processing_info)
                
                progress_bar.progress(75)
                status_text.text("✅ 仕訳データ抽出完了！")
                progress_bar.progress(100)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # 結果表示（エラー情報がある場合は警告表示）
                if processing_info.get("error"):
                    st.warning(f"⚠️ 処理は完了しましたが、一部でエラーが発生しました: {processing_info['error']}")
                else:
                    st.success(f"🎉 変換が完了しました！処理時間: {processing_time:.1f}秒")
                
                # プログレスバーを削除
                progress_bar.empty()
                status_text.empty()
                
                # 結果サマリー
                col_result1, col_result2, col_result3 = st.columns(3)
                with col_result1:
                    st.metric("抽出エントリ数", len(df))
                with col_result2:
                    st.metric("処理時間", f"{processing_time:.1f}秒")
                with col_result3:
                    # API費用概算表示（トークンベース）
                    if processing_info.get("cost_jpy", 0) > 0:
                        # 最新レート情報を取得して表示
                        try:
                            current_rate = config.get_current_usd_to_jpy_rate()
                            rate_info = f"為替レート: {current_rate:.2f} JPY/USD"
                        except:
                            rate_info = "為替レート: 取得失敗"
                            
                        st.metric(
                            "API費用概算", 
                            f"¥{processing_info['cost_jpy']:.2f}",
                            help=f"⚠️ 概算値です。実際の請求と異なる場合があります。\n\n計算値: ${processing_info['cost_usd']:.4f} USD\n{rate_info}\nトークン使用量×API単価で計算\n\n※ 為替レート変動や請求タイミングにより実際の費用と差が生じる可能性があります"
                        )
                    else:
                        # フォールバック: モックデータまたは費用計算失敗時
                        pages_processed = processing_info.get('pages_processed', max(1, len(df) // 5))
                        estimated_cost_jpy = pages_processed * 15  # 1ページあたり約15円の概算
                        
                        st.metric(
                            "API費用概算", 
                            f"¥{estimated_cost_jpy:.0f}",
                            help=f"概算: {pages_processed}ページ × ¥15/ページ\n※実際の費用はトークン数により変動"
                        )
                
                # データプレビュー
                if not df.empty:
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
                    
                    # データ表示
                    st.dataframe(
                        df.head(display_count), 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    if len(df) > display_count:
                        st.info(f"表示: {display_count}件 / 全{len(df)}件")
                else:
                    st.warning("⚠️ 抽出結果が空でした。PDFの内容をご確認ください。")
                
                # 抽出状況チェックとダウンロードボタン
                st.divider()
                entries_extracted = len(df)
                
                if entries_extracted == 0:
                    st.warning("⚠️ **抽出が未実行です**。『🚀 仕訳データ抽出開始』を押してください")
                    st.info("📝 CSVダウンロードは抽出完了後に有効になります")
                    
                    # 無効化されたダウンロードボタン
                    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                    with col_dl2:
                        st.button(
                            "📥 45列CSVをダウンロード (無効)",
                            disabled=True,
                            use_container_width=True,
                            help="抽出完了後に有効になります"
                        )
                else:
                    # 有効なダウンロードボタン
                    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                    with col_dl2:
                        download_filename = f"{Path(uploaded_file.name).stem}_mjs45_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        st.download_button(
                            label="📥 ミロク取込45列CSV をダウンロード",
                            data=csv_bytes,
                            file_name=download_filename,
                            mime="text/csv",
                            use_container_width=True,
                            type="secondary",
                            help="ミロク会計システムに直接取り込み可能な45列形式のCSVファイル"
                        )
                
            except Exception as e:
                st.error(f"💥 変換に失敗しました: {str(e)}")
                logger.error(f"PDF processing error: {e}", exc_info=True)
                
                # エラー時でも処理情報を初期化
                processing_info = {
                    "cost_usd": 0.0,
                    "cost_jpy": 0.0,
                    "processing_time": 0.0,
                    "pages_processed": 0,
                    "entries_extracted": 0
                }
                
                # エラー詳細（デバッグモード時）
                if config.DEBUG_MODE:
                    with st.expander("🔍 エラー詳細（デバッグ情報）"):
                        st.code(str(e))

# 使用方法とヒント
st.divider()
with st.expander("📖 使用方法とヒント"):
    st.markdown("""
    ### 📋 使用手順
    1. **PDFファイル選択**: 仕訳データが含まれるPDFファイルを選択
    2. **抽出開始**: 「仕訳データ抽出開始」ボタンをクリック
    3. **結果確認**: 抽出されたデータをプレビューで確認
    4. **CSV出力**: 「CSVファイルをダウンロード」でファイル保存
    
    ### 💡 処理のポイント
    - **分割処理**: 大きなPDFは5ページずつ分割して効率的に処理
    - **高精度AI**: Claude Sonnet 4.0の視覚認識で正確な仕訳抽出
    - **日本語対応**: 日本の会計基準に対応した仕訳形式で出力
    
    ### ⚠️ 注意事項
    - **処理時間**: 大きなファイルや複雑なレイアウトは時間がかかります
    - **データ精度**: 手書きや画質の悪いPDFは抽出精度が下がる場合があります
    - **プライバシー**: アップロードファイルは一時的にメモリ上で処理され、サーバーには保存されません
    """)

# フッター
st.divider()
st.caption("📊 PDF仕訳抽出システム | Powered by Claude Sonnet 4.0 | Built with Streamlit")
