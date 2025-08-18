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

# メインコンテンツ
st.title("📊 PDF仕訳抽出システム")
st.markdown("### 📄 PDFファイルから会計仕訳データを自動抽出してCSVで出力")

# アップロード部分
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📁 PDFファイルを選択してください",
        type=["pdf"],
        help="5ページずつ分割処理されます。大きなファイルは時間がかかる場合があります。"
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
    
    # 変換ボタン
    st.divider()
    col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
    
    with col_button2:
        convert_clicked = st.button(
            "🚀 仕訳データ抽出開始", 
            type="primary",
            use_container_width=True,
            help="Claude Sonnet 4.0を使用してPDFから仕訳データを抽出します"
        )
    
    # 処理実行
    if convert_clicked:
        with st.spinner("🔄 Claude Sonnet 4.0で仕訳データを抽出中..."):
            try:
                start_time = datetime.now()
                
                # プログレスバー表示
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📄 PDFファイルを分析中...")
                progress_bar.progress(25)
                
                # メイン処理
                df, csv_bytes = process_pdf_to_csv(uploaded_file)
                
                progress_bar.progress(75)
                status_text.text("✅ 仕訳データ抽出完了！")
                progress_bar.progress(100)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # 結果表示
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
                    if not df.empty and '金額' in df.columns:
                        total_amount = df['金額'].astype(str).str.replace(',', '').astype(float).sum()
                        st.metric("合計金額", f"¥{total_amount:,.0f}")
                    else:
                        st.metric("合計金額", "計算不可")
                
                # データプレビュー
                if not df.empty:
                    st.divider()
                    st.subheader("📋 抽出データプレビュー")
                    
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
                
                # ダウンロードボタン
                st.divider()
                col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                with col_dl2:
                    download_filename = f"{Path(uploaded_file.name).stem}_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.download_button(
                        label="📥 CSVファイルをダウンロード",
                        data=csv_bytes,
                        file_name=download_filename,
                        mime="text/csv",
                        use_container_width=True,
                        type="secondary"
                    )
                
            except Exception as e:
                st.error(f"💥 変換に失敗しました: {str(e)}")
                logger.error(f"PDF processing error: {e}", exc_info=True)
                
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
