#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON抽出ガード機能の簡易テスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import unittest
from unittest.mock import patch
from utils.json_guard import (
    parse_5cols_json,
    extract_json_array_str,
    get_fallback_entry,
    JSONExtractionTimeout
)

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
        """無効なJSON形式でのフォールバック"""
        response = "これはJSONではありません"

        # parse_5cols_json は例外を投げず、フォールバックを返す
        result = parse_5cols_json(response)
        self.assertEqual(len(result), 1)
        self.assertIn("OCR注意", result[0]["摘要"])

    def test_malformed_json(self):
        """不正なJSON形式でのフォールバック"""
        response = '[{"伝票日付":"2025/2/10","借貸区分":"借方",]'  # 不正な形式

        # parse_5cols_json は例外を投げず、フォールバックを返す
        result = parse_5cols_json(response)
        self.assertEqual(len(result), 1)
        self.assertIn("OCR注意", result[0]["摘要"])
    
    def test_fallback_entry(self):
        """フォールバックエントリの生成"""
        fallback = get_fallback_entry("テストエラー")

        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]["金額"], 1)  # フォールバックは金額=1（空出力防止）
        self.assertIn("テストエラー", fallback[0]["摘要"])
        self.assertIn("【OCR注意:目視確認推奨】", fallback[0]["摘要"])
    
    def test_extract_json_array_str(self):
        """JSON配列文字列の抽出テスト"""
        text = '前置き [{"test": "value"}] 後置き'
        result = extract_json_array_str(text)
        self.assertEqual(result, '[{"test": "value"}]')
    
    def test_bracket_completion_missing_end(self):
        """角括弧補完テスト：終了角括弧不足"""
        text = '[{"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":1000,"摘要":"テスト"}'
        result = extract_json_array_str(text)
        # 終了角括弧が補完されること
        self.assertTrue(result.endswith(']'))
        # パースできること
        data = json.loads(result)
        self.assertEqual(len(data), 1)
    
    def test_bracket_completion_missing_start(self):
        """角括弧補完テスト：開始角括弧不足"""
        text = '{"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":1000,"摘要":"テスト"}]'
        result = extract_json_array_str(text)
        # 開始角括弧が補完されること
        self.assertTrue(result.startswith('['))
        # パースできること
        data = json.loads(result)
        self.assertEqual(len(data), 1)
    
    def test_various_code_fences(self):
        """様々なコードフェンス形式のテスト"""
        test_cases = [
            '```JSON\n[{"test": "value"}]\n```',
            '~~~json\n[{"test": "value"}]\n~~~',
            '```json\n[{"test": "value"}]\n```',
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = extract_json_array_str(text)
                self.assertEqual(result, '[{"test": "value"}]')
    
    def test_robust_json_extraction(self):
        """頑健なJSON抽出テスト（実LLM出力模擬）"""
        # 実際のLLM出力によくある形式
        text = '''
        以下が抽出結果です：
        
        [
            {
                "伝票日付": "2025/2/10",
                "借貸区分": "借方", 
                "科目名": "現金",
                "金額": 1000,
                "摘要": "テスト取引; 借方:現金"
            },
            {
                "伝票日付": "2025/2/10",
                "借貸区分": "貸方",
                "科目名": "売上高", 
                "金額": 1000,
                "摘要": "テスト取引; 貸方:売上高"
            }
        ]
        
        以上が結果です。
        '''
        
        result = parse_5cols_json(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["科目名"], "現金")
        self.assertEqual(result[1]["借貸区分"], "貸方")

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


class TestLinearScanExtraction(unittest.TestCase):
    """線形スキャン方式のJSON抽出テスト（P0対応）"""

    def test_nested_json_arrays(self):
        """ネストした配列を含むJSON抽出"""
        text = '''[
            {
                "伝票日付": "2025/2/10",
                "借貸区分": "借方",
                "科目名": "現金",
                "金額": 1000,
                "摘要": "配列テスト: [内部配列]",
                "metadata": ["tag1", "tag2"]
            }
        ]'''
        result = extract_json_array_str(text)
        data = json.loads(result)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["科目名"], "現金")
        self.assertIsInstance(data[0]["metadata"], list)

    def test_json_with_escaped_quotes(self):
        """エスケープされた引用符を含むJSON"""
        text = r'[{"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":1000,"摘要":"テスト\"引用符\""}]'
        result = extract_json_array_str(text)
        data = json.loads(result)
        self.assertEqual(len(data), 1)
        self.assertIn('引用符', data[0]["摘要"])

    def test_json_with_brackets_in_string(self):
        """文字列リテラル内に括弧を含むJSON"""
        text = '[{"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":1000,"摘要":"摘要[括弧付き]"}]'
        result = extract_json_array_str(text)
        data = json.loads(result)
        self.assertEqual(len(data), 1)
        self.assertIn('[括弧付き]', data[0]["摘要"])

    def test_multiple_json_arrays_first_valid(self):
        """複数のJSON配列があり、最初が有効な場合"""
        text = '''
        [{"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":1000,"摘要":"最初"}]
        [{"伝票日付":"2025/2/11","借貸区分":"貸方","科目名":"売上","金額":2000,"摘要":"2番目"}]
        '''
        result = extract_json_array_str(text)
        data = json.loads(result)
        # 最初の配列が抽出されること
        self.assertEqual(data[0]["摘要"], "最初")

    def test_multiple_json_arrays_first_invalid(self):
        """複数のJSON配列があり、最初が無効な場合"""
        text = '''
        [invalid json here]
        [{"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":1000,"摘要":"2番目が正しい"}]
        '''
        result = extract_json_array_str(text)
        data = json.loads(result)
        # 2番目の有効な配列が抽出されること
        self.assertEqual(data[0]["摘要"], "2番目が正しい")

    def test_large_json_extraction(self):
        """大きなJSON配列の抽出（パフォーマンステスト）"""
        # 100エントリの配列を生成
        entries = []
        for i in range(100):
            entries.append({
                "伝票日付": "2025/2/10",
                "借貸区分": "借方" if i % 2 == 0 else "貸方",
                "科目名": f"科目{i}",
                "金額": 1000 * (i + 1),
                "摘要": f"テスト取引{i}"
            })
        large_json = json.dumps(entries, ensure_ascii=False)

        start = time.time()
        result = extract_json_array_str(large_json)
        elapsed = time.time() - start

        data = json.loads(result)
        self.assertEqual(len(data), 100)
        # 2秒未満で完了すること（タイムアウトより短い）
        self.assertLess(elapsed, 2.0)

    def test_timeout_protection(self):
        """タイムアウト保護のテスト（短いタイムアウト設定）"""
        # 非常に大きなテキストを生成（50万文字の文字列リテラル）
        # これは線形スキャンでも時間がかかる
        huge_text = '[{"data": "' + "x" * 500000 + '"}]'

        # タイムアウトが発生すること（0.01秒で設定）
        with self.assertRaises(JSONExtractionTimeout):
            extract_json_array_str(huge_text, timeout_seconds=0.01)

    def test_deeply_nested_arrays(self):
        """深くネストした配列の処理"""
        text = '[{"data": [[["深いネスト"]]], "伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":1000,"摘要":"ネストテスト"}]'
        result = extract_json_array_str(text)
        data = json.loads(result)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["科目名"], "現金")

    def test_json_with_newlines_in_strings(self):
        """文字列内に改行を含むJSON"""
        text = r'[{"伝票日付":"2025/2/10","借貸区分":"借方","科目名":"現金","金額":1000,"摘要":"行1\n行2\n行3"}]'
        result = extract_json_array_str(text)
        data = json.loads(result)
        self.assertEqual(len(data), 1)
        self.assertIn('\\n', result)  # エスケープされた改行が保持されること

    def test_fallback_returns_single_entry(self):
        """フォールバック時に必ず1エントリ返すこと"""
        invalid_text = "これは完全に無効なテキストです。JSONは含まれません。"

        # parse_5cols_json は例外を投げずにフォールバックを返す
        result = parse_5cols_json(invalid_text)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["借貸区分"], "借方")
        self.assertIn("OCR注意", result[0]["摘要"])

    def test_performance_no_catastrophic_backtracking(self):
        """正規表現のcatastrophic backtrackingが発生しないこと"""
        # 大量の括弧を含む病的なテキスト（従来の正規表現では固まる）
        pathological = "[" * 1000 + "test" + "]" * 500

        start = time.time()
        try:
            # エラーになっても良いが、タイムアウト以内に完了すること
            extract_json_array_str(pathological, timeout_seconds=1.0)
        except (ValueError, JSONExtractionTimeout):
            pass  # エラーはOK
        elapsed = time.time() - start

        # 1.5秒以内に完了すること（タイムアウト+マージン）
        self.assertLess(elapsed, 1.5)

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