#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 出力系
OUT_DIR="$ROOT/output"
BACKUP_DIR="$OUT_DIR/backups"
TMP_DIR="$OUT_DIR/tmp"

echo "Cleaning up test artifacts and temporary files..."

# 1) デモ/テスト由来のCSV類を削除
find "$OUT_DIR" -maxdepth 1 -type f \( \
  -name '*_demo.csv' -o -name '*_fixed_demo.csv' -o -name '*_dedup_demo.csv' -o -name '*_test*.csv' \
\) -print -delete 2>/dev/null || true

# 2) 一時分割や中間成果物
[ -d "$TMP_DIR" ] && find "$TMP_DIR" -type f -mtime +1 -print -delete 2>/dev/null || true

# 3) 古いバックアップ（直近7世代だけ残す）
if [ -d "$BACKUP_DIR" ]; then
  ls -1t "$BACKUP_DIR"/*.pdf 2>/dev/null | tail -n +8 | xargs -r rm -f || true
fi

# 4) ログのローテーション（7日超は削除）
[ -d "$ROOT/logs" ] && find "$ROOT/logs" -type f -mtime +7 -print -delete 2>/dev/null || true

# 5) Python cache files
find "$ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$ROOT" -name "*.pyc" -delete 2>/dev/null || true

echo "Cleanup done."