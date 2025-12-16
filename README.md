# 🤖 accountr-ai

AI-Powered Accounting Journal Extraction System

Claude Sonnet 4.5を使用してPDFファイルから会計仕訳データを自動抽出し、CSV形式で出力するStreamlitアプリケーション。

## 🚀 特徴

- **高精度AI処理**: Claude Sonnet 4.5の視覚認識による正確な仕訳データ抽出
- **自動分割処理**: 大きなPDFを5ページずつ分割して効率的に処理
- **リアルタイム処理**: Webブラウザから直接PDF操作、即座にCSV出力
- **日本語完全対応**: 日本の会計基準に対応した仕訳形式
- **プライバシー保護**: アップロードファイルはメモリ上でのみ処理

## 📋 システム要件

- Python 3.8+
- Claude Sonnet 4.5 API キー (Anthropic)
- 1GB RAM以上推奨

## 🛠️ ローカル開発環境セットアップ

### 1. リポジトリクローン & 依存関係インストール
```bash
git clone [repository-url]
cd simple_ocr
pip install -r requirements.txt
```

### 2. 環境変数設定
`.env`ファイルを作成し、以下を設定：
```env
# Anthropic Claude API
ANTHROPIC_API_KEY=sk-ant-api03-your-api-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
ANTHROPIC_MAX_TOKENS=64000
ANTHROPIC_BETA_HEADERS=context-1m-2025-08-07

# 処理設定
USE_MOCK_DATA=false
DEBUG_MODE=false
PAGES_PER_SPLIT=5
```

### 3. ローカル実行
```bash
streamlit run app.py
```

## ☁️ Streamlit Cloud デプロイメント

### 前提条件
- GitHubアカウント
- Anthropic Claude APIキー

### 1. GitHub リポジトリ作成
```bash
# リポジトリ作成
git init
git add .
git commit -m "Initial commit: accountr-ai"
git branch -M main
git remote add origin https://github.com/[username]/accountr-ai.git
git push -u origin main
```

### 2. Streamlit Cloud デプロイ
1. [share.streamlit.io](https://share.streamlit.io) にログイン
2. "New app" をクリック
3. GitHub リポジトリを選択
4. **Branch**: main
5. **Main file path**: accountr-ai/app.py
6. **App URL**: 任意のURLを設定

### 3. 🔐 Secrets設定（重要）
**Settings** → **Secrets** で以下を設定：
```toml
ANTHROPIC_API_KEY = "sk-ant-api03-your-actual-api-key-here"
```

### 4. Advanced Settings（オプション）
```toml
USE_MOCK_DATA = false
DEBUG_MODE = false
PAGES_PER_SPLIT = 5
```

### 5. デプロイ実行
- **Deploy!** ボタンをクリック
- 自動ビルド開始（約2-3分）
- デプロイ完了後、URLでアクセス可能

## 📁 プロジェクト構造

```
simple_ocr/
├── app.py                 # Streamlit メインアプリケーション
├── backend_processor.py   # PDF処理バックエンド
├── pdf_extractor.py      # Claude API連携・PDF抽出ロジック
├── config.py             # 設定管理
├── requirements.txt      # 依存関係
├── .env                  # 環境変数（ローカル用）
├── .env.production       # 本番環境用設定テンプレート
└── README.md             # このドキュメント
```

## 🔧 設定オプション

| 環境変数 | 説明 | デフォルト |
|---------|------|-----------|
| `ANTHROPIC_API_KEY` | Claude API キー | 必須 |
| `ANTHROPIC_MODEL` | 使用モデル | claude-sonnet-4-0 |
| `ANTHROPIC_MAX_TOKENS` | 最大トークン数 | 64000 |
| `USE_MOCK_DATA` | テストモード | false |
| `PAGES_PER_SPLIT` | PDF分割単位 | 5 |
| `DEBUG_MODE` | デバッグログ出力 | false |

## 📊 使用方法

1. **PDFアップロード**: ブラウザでファイル選択
2. **処理実行**: 「仕訳データ抽出開始」ボタンクリック
3. **結果確認**: 抽出データをプレビュー表示
4. **CSV出力**: 「CSVファイルをダウンロード」で保存

## 🔍 トラブルシューティング

### API関連エラー
- APIキーが正しく設定されているか確認
- Claude API の利用制限に達していないか確認
- インターネット接続を確認

### PDF処理エラー
- PDFファイルが暗号化されていないか確認
- ファイルサイズが適切か確認（推奨: 50MB以下）
- PDF内容が画像形式でなく、テキスト・表形式か確認

### メモリエラー
- 大きなPDFは分割処理されるが、極端に大きい場合は分割
- ブラウザキャッシュをクリア

## 💰 API使用料金目安

Claude Sonnet 4.5 料金（2025年）：
- 入力: $3 per 1M tokens
- 出力: $15 per 1M tokens

**処理コスト例**：
- 5ページPDF: 約 $0.05-0.15
- 20ページPDF: 約 $0.20-0.50
- 100ページPDF: 約 $1.00-2.50

## 🛡️ セキュリティ

- APIキーは環境変数・シークレットで管理
- アップロードファイルはメモリ上でのみ処理
- ログ出力でAPIキーをマスク
- HTTPS通信でデータ保護

## 📄 ライセンス

このプロジェクトは適切なライセンスの下で提供されています。

## 🤝 サポート

問題や質問がある場合は、GitHub Issues をご利用ください。