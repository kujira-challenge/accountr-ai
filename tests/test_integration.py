#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合テストスクリプト - 5カラムJSON→45列CSV変換テスト
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# プロジェクトのルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mjs_converter import fivejson_to_mjs45, MJSConverter

def test_integration():
    """統合テスト実行"""
    print("=" * 60)
    print("🧪 統合テスト: 5カラムJSON → 45列MJS CSV変換")
    print("=" * 60)
    
    # テストファイルパス
    test_dir = Path(__file__).parent
    sample_json = test_dir / "sample_5col.json"
    dummy_codes = test_dir / "dummy_account_codes.csv"
    output_csv = test_dir / "test_output_mjs45.csv"
    log_file = test_dir / "test_conversion.log"
    
    # 入力ファイル存在確認
    if not sample_json.exists():
        print(f"❌ サンプルJSONが見つかりません: {sample_json}")
        return False
    
    if not dummy_codes.exists():
        print(f"❌ ダミーコード表が見つかりません: {dummy_codes}")
        return False
    
    print(f"📄 サンプルJSON: {sample_json}")
    print(f"📊 コード表CSV: {dummy_codes}")
    print(f"📝 出力CSV: {output_csv}")
    
    try:
        # 変換実行
        print("\n🔄 変換実行中...")
        fivejson_to_mjs45(
            str(sample_json),
            str(dummy_codes),
            str(output_csv),
            str(log_file)
        )
        
        # 結果検証
        print("✅ 変換完了! 結果を検証中...")
        
        # 1. 出力ファイル存在確認
        if not output_csv.exists():
            print("❌ 出力CSVファイルが作成されていません")
            return False
        
        # 2. CSVを読み込んで検証
        df = pd.read_csv(output_csv, encoding='utf-8-sig')
        
        # 3. ヘッダー確認（45列完全一致）
        expected_columns = MJSConverter.MJS_45_COLUMNS
        if list(df.columns) != expected_columns:
            print(f"❌ ヘッダーが45列と一致しません")
            print(f"期待値: {len(expected_columns)}列")
            print(f"実際: {len(df.columns)}列")
            return False
        print(f"✅ ヘッダー: 45列完全一致")
        
        # 4. データ行数確認
        expected_rows = 3  # 借方2 + 貸方1
        if len(df) != expected_rows:
            print(f"❌ データ行数が一致しません (期待: {expected_rows}, 実際: {len(df)})")
            return False
        print(f"✅ データ行数: {len(df)}行")
        
        # 5. 金額確認
        expected_amounts = [49000, 33500, 82500]
        actual_amounts = df['金額'].astype(int).tolist()
        if actual_amounts != expected_amounts:
            print(f"❌ 金額が一致しません (期待: {expected_amounts}, 実際: {actual_amounts})")
            return False
        print(f"✅ 金額: {actual_amounts}")
        
        # 6. 借方/貸方コード設定確認
        debit_codes = df['（借）科目ｺｰﾄﾞ'].fillna('').tolist()
        credit_codes = df['（貸）科目ｺｰﾄﾞ'].fillna('').tolist()
        
        # 借方2行はコード設定、貸方列は空
        if not (debit_codes[0] and debit_codes[1] and not debit_codes[2]):
            print(f"❌ 借方科目コードの設定が不正: {debit_codes}")
            return False
            
        # 貸方1行はコード設定、借方列は空
        if not (not credit_codes[0] and not credit_codes[1] and credit_codes[2]):
            print(f"❌ 貸方科目コードの設定が不正: {credit_codes}")
            return False
            
        print(f"✅ 科目コード補完: 借方={[c for c in debit_codes if c]}, 貸方={[c for c in credit_codes if c]}")
        
        # 7. 摘要の共通部分確認
        common_part = "退居振替; オーナー: 飯島えり子; 物件名: ルベール武蔵関; 号室: 101; 契約者名: 所 厚作"
        for i, memo in enumerate(df['摘要'].tolist()):
            if not memo.startswith(common_part):
                print(f"❌ 摘要{i+1}に共通部分が継承されていません: {memo[:50]}...")
                return False
        print(f"✅ 共通摘要継承: 全行に適用済み")
        
        # 8. 借方/貸方合計確認
        debit_total = sum(df[df['（借）科目ｺｰﾄﾞ'].fillna('') != '']['金額'])
        credit_total = sum(df[df['（貸）科目ｺｰﾄﾞ'].fillna('') != '']['金額'])
        
        if debit_total != credit_total:
            print(f"❌ 借貸バランス不一致: 借方={debit_total}, 貸方={credit_total}")
            return False
        print(f"✅ 借貸バランス: 借方={debit_total}, 貸方={credit_total}")
        
        print(f"\n🎉 全ての検証に合格しました!")
        
        # サンプル出力表示
        print(f"\n📋 出力CSVサンプル (最初の3列):")
        sample_cols = ['伝票日付', '（借）科目ｺｰﾄﾞ', '（貸）科目ｺｰﾄﾞ', '金額', '摘要']
        display_df = df[sample_cols]
        print(display_df.to_string(index=False, max_colwidth=50))
        
        return True
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 一時ファイルクリーンアップ
        for temp_file in [output_csv, log_file]:
            if temp_file.exists():
                temp_file.unlink()
                print(f"🧹 一時ファイル削除: {temp_file.name}")

def main():
    """メイン関数"""
    success = test_integration()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 統合テスト成功!")
        exit_code = 0
    else:
        print("❌ 統合テスト失敗!")
        exit_code = 1
    print(f"{'='*60}")
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)