#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON抽出ガード機能の簡易テスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch
from utils.json_guard import parse_5cols_json, extract_json_array_str, get_fallback_entry

class TestJsonGuard(unittest.TestCase):
    """JSON抽出ガード機能のテストケース"""
    
    def test_normal_json_extraction(self):
        """正常系：きれいな5カラムJSONの抽出"""
        response = """
        [
            {"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":1000,"摘要":"テスト取引; 借方:現金"},
            {"伝票日付":"2025/2/10","借貸区分":"貸方","科目名":"売上高","金額":1000,"摘要":"テスト取引; 貸方:売上高"}
        ]
        """
        
        result = parse_5cols_json(response)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["科目名"], "現金")
        self.assertEqual(result[1]["借貸区分"], "貸方")
        self.assertEqual(result[0]["金額"], 1000)
    
    def test_json_with_code_blocks(self):
        """コードブロック付きJSONの抽出"""
        response = """
        説明テキスト
        ```json
        [
            {"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"預金","金額":5000,"摘要":"テスト; 借方:預金"}
        ]
        ```
        後続テキスト
        """
        
        result = parse_5cols_json(response)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["科目名"], "預金")
    
    def test_json_with_prefix_suffix(self):
        """前置き・後置きテキスト付きJSONの抽出"""
        response = """
        以下が結果です：
        [{"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":2000,"摘要":"テスト"}]
        以上です。
        """
        
        result = parse_5cols_json(response)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["金額"], 2000)
    
    def test_missing_required_columns(self):
        """必須カラム不足時の自動補完"""
        response = """
        [{"伝票日付":"2025/2/10","科目名":"現金","金額":1000}]
        """
        
        result = parse_5cols_json(response)
        self.assertEqual(len(result), 1)
        # 不足カラムが空文字で補完されること
        self.assertEqual(result[0]["借貸区分"], "")
        self.assertEqual(result[0]["摘要"], "")
    
    def test_invalid_amount_conversion(self):
        """金額の型変換テスト"""
        response = """
        [{"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":"1,500","摘要":"テスト"}]
        """
        
        result = parse_5cols_json(response)
        self.assertEqual(result[0]["金額"], 1500)  # カンマが除去され整数に変換
    
    def test_invalid_json_format(self):
        """無効なJSON形式でのエラー"""
        response = "これはJSONではありません"
        
        with self.assertRaises(ValueError):
            parse_5cols_json(response)
    
    def test_malformed_json(self):
        """不正なJSON形式でのエラー"""
        response = '[{"伝票日付":"2025/2/10","借貸区分":"借方",]'  # 不正な形式
        
        with self.assertRaises(ValueError):
            parse_5cols_json(response)
    
    def test_fallback_entry(self):
        """フォールバックエントリの生成"""
        fallback = get_fallback_entry("テストエラー")
        
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]["金額"], 0)
        self.assertIn("テストエラー", fallback[0]["摘要"])
        self.assertIn("【OCR注意:目視確認推奨】", fallback[0]["摘要"])
    
    def test_extract_json_array_str(self):
        """JSON配列文字列の抽出テスト"""
        text = "前置き [{'test': 'value'}] 後置き"
        result = extract_json_array_str(text)
        self.assertEqual(result, "[{'test': 'value'}]")

class TestIntegrationWithMock(unittest.TestCase):
    """LLM呼び出しをモックした統合テスト"""
    
    @patch('pdf_extractor.ProductionPDFExtractor._call_claude_api')
    def test_json_guard_integration(self, mock_api_call):
        """JSON抽出ガードの統合テスト（モック使用）"""
        
        # モックレスポンスの設定
        mock_response = """
        以下が抽出結果です：
        [
            {"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":1000,"摘要":"統合テスト; 借方:現金"},
            {"伝票日付":"2025/2/10","借貸区分":"貸方","科目名":"売上高","金額":1000,"摘要":"統合テスト; 貸方:売上高"}
        ]
        以上です。
        """
        mock_api_call.return_value = (mock_response, 0.01)  # (response, cost)
        
        # 実際のparse_5cols_json関数をテスト
        result = parse_5cols_json(mock_response)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["科目名"], "現金")
        self.assertEqual(result[1]["借貸区分"], "貸方")

def run_tests():
    """テストの実行"""
    print("=== JSON抽出ガード機能 簡易テスト ===")
    
    # テストスイートの実行
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果サマリー
    print(f"\n=== テスト結果 ===")
    print(f"実行: {result.testsRun}件")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}件")
    print(f"失敗: {len(result.failures)}件")
    print(f"エラー: {len(result.errors)}件")
    
    if result.failures:
        print("\n失敗したテスト:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"- {test}: {error_msg}")
    
    if result.errors:
        print("\nエラーが発生したテスト:")
        for test, traceback in result.errors:
            error_lines = traceback.split('\n')
            error_msg = error_lines[-2] if len(error_lines) > 1 else str(test)
            print(f"- {test}: {error_msg}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)