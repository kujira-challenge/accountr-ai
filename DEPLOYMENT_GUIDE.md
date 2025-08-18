# 🚀 accountr-ai Streamlit Cloud デプロイガイド

このガイドでは、accountr-aiをStreamlit Cloudにデプロイするための完全な手順を説明します。

## 📋 前提条件

- ✅ GitHubアカウント
- ✅ Anthropic Claude APIキー ([console.anthropic.com](https://console.anthropic.com)で取得)
- ✅ このプロジェクトのファイル一式

## 🔧 デプロイ前の準備

### 1. 必要ファイルの確認
以下のファイルが`accountr-ai/`ディレクトリに存在することを確認：

```
accountr-ai/
├── app.py                     ✅ メインアプリケーション
├── backend_processor.py       ✅ バックエンド処理
├── pdf_extractor.py          ✅ PDF抽出ロジック
├── config.py                 ✅ 設定管理（Streamlit Secrets対応済み）
├── requirements.txt          ✅ 依存関係
├── packages.txt              ✅ システム依存関係
├── .gitignore                ✅ Git除外設定
├── .streamlit/
│   └── secrets.toml          ✅ ローカル開発用（GitHubにコミットされない）
└── README.md                 ✅ ドキュメント
```

### 2. APIキーの準備
Anthropic Claude APIキーを準備：
- 形式：`sk-ant-api03-...` で始まる文字列
- 取得先：https://console.anthropic.com

## 📂 GitHubリポジトリ作成

### 1. GitHubでリポジトリ作成
1. GitHubにログイン
2. "New repository"をクリック
3. リポジトリ名：`accountr-ai`（推奨）
4. Public/Privateを選択
5. "Create repository"をクリック

### 2. コードをGitHubにプッシュ
```bash
# accountr-aiディレクトリで実行
cd accountr-ai

# Gitリポジトリ初期化
git init

# ファイルをステージング
git add .

# 初回コミット
git commit -m "Initial commit: accountr-ai Streamlit app"

# メインブランチ作成
git branch -M main

# リモートリポジトリ追加（[username]を自分のユーザー名に変更）
git remote add origin https://github.com/[username]/accountr-ai.git

# GitHubにプッシュ
git push -u origin main
```

## ☁️ Streamlit Cloud デプロイ

### 1. Streamlit Cloudでアプリ作成
1. [share.streamlit.io](https://share.streamlit.io) にアクセス
2. GitHubアカウントでログイン
3. "New app"をクリック
4. 以下の設定を入力：
   - **Repository**: `[username]/accountr-ai`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: 任意のURL名（例：`my-accountr-ai`）

### 2. 🔐 Secrets設定（最重要）
1. "Advanced settings"をクリック
2. "Secrets"タブを選択
3. 以下の内容を入力：

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-your-actual-api-key-here"
```

⚠️ **注意**: 
- APIキーは実際の値に置き換えてください
- ダブルクォートを忘れずに
- 他の設定項目は不要（デフォルト値が使用されます）

### 3. デプロイ実行
1. "Deploy!"ボタンをクリック
2. ビルドプロセス開始（2-3分程度）
3. 成功時：緑色の"Your app is live!"メッセージ
4. エラー時：ログを確認してトラブルシューティング

## ✅ デプロイ後の確認

### 1. アプリの動作確認
1. デプロイされたURLにアクセス
2. サイドバーで"✅ Claude API接続準備完了"を確認
3. PDFファイルアップロードをテスト
4. 仕訳データ抽出の動作確認

### 2. エラーチェック
もし以下のエラーが表示される場合：

#### "❌ Claude APIキーが未設定"
→ Streamlit CloudのSecretsでAPIキーを再確認

#### "🚫 Claude APIキーが設定されていません"
→ アプリを再起動：Streamlit Cloud管理画面で"Reboot app"

#### ビルドエラー
→ requirements.txtやpackages.txtの内容を確認

## 🔧 設定カスタマイズ（オプション）

必要に応じてSecretsに追加設定：

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-your-actual-api-key-here"
USE_MOCK_DATA = false
DEBUG_MODE = false
PAGES_PER_SPLIT = 5
MAX_FILE_SIZE_MB = 50
```

## 🔄 アプリの更新

コードを変更する場合：

```bash
# 変更をコミット
git add .
git commit -m "Update: 変更内容の説明"
git push

# Streamlit Cloudで自動的に再デプロイされます
```

## 🐛 トラブルシューティング

### よくある問題と解決法

1. **APIキーエラー**
   - Secretsの設定を再確認
   - APIキーの形式確認（sk-ant-api03-で始まる）
   - Anthropicアカウントの残高確認

2. **ビルドエラー**
   - GitHubリポジトリの内容確認
   - requirements.txtの依存関係確認
   - packages.txtのシステム依存関係確認

3. **パフォーマンス問題**
   - ファイルサイズ制限確認（推奨50MB以下）
   - PDF分割設定調整（PAGES_PER_SPLIT）

4. **アクセス権限エラー**
   - GitHubリポジトリがPublicか確認
   - Streamlit CloudのGitHub連携確認

## 📞 サポート

問題が解決しない場合：
1. GitHub Issuesで報告
2. Streamlit Community Forumで質問
3. Anthropic APIドキュメント確認

---

🎉 **デプロイ完了！** 
これでaccountr-aiがStreamlit Cloudで稼働し、世界中からアクセス可能になりました。