#!/usr/bin/env bash
# cleanup_outputs.sh - Enhanced cleanup script for accounting journal system
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Accounting Journal System Cleanup ==="
echo "Root directory: $ROOT"

# Clean test and demo CSV files
echo "1. Cleaning test and demo files..."
find "$ROOT" -name "*_demo.csv" -type f -delete 2>/dev/null || true
find "$ROOT" -name "*_test.csv" -type f -delete 2>/dev/null || true
echo "   Test and demo CSV files removed"

# Clean temporary output directories  
echo "2. Cleaning temporary directories..."
[ -d "$ROOT/output/tmp" ] && rm -rf "$ROOT/output/tmp" && echo "   output/tmp removed"
[ -d "$ROOT/logs" ] && rm -rf "$ROOT/logs" && echo "   logs/ removed"

# Backup rotation (keep only 3 most recent)
echo "3. Managing backup rotation..."
if [ -d "$ROOT/output" ]; then
    # Remove old backup directories, keep only 3 most recent
    ls -dt "$ROOT/output"/backup_* 2>/dev/null | tail -n +4 | xargs -r rm -rf
    backup_count=$(ls -d "$ROOT/output"/backup_* 2>/dev/null | wc -l)
    echo "   Backup rotation complete (${backup_count} directories remaining)"
else
    echo "   No output directory found"
fi

# Clean Python cache files
echo "4. Cleaning Python cache..."
find "$ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$ROOT" -name "*.pyc" -delete 2>/dev/null || true
find "$ROOT" -name "*.pyo" -delete 2>/dev/null || true
echo "   Python cache files removed"

# Summary
echo "=== Cleanup Summary ==="
if [ -d "$ROOT/output" ]; then
    output_size=$(du -sh "$ROOT/output" 2>/dev/null | cut -f1 || echo "unknown")
    echo "Output directory size: $output_size"
fi
echo "Cleanup completed successfully!"