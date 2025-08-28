#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ミロク取込45列CSV変換器
5カラムJSONをミロク取込45列CSVに変換し、科目名から科目コードを補完する
"""

import json
import csv
import re
import unicodedata
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
import pandas as pd

# ログ設定
logger = logging.getLogger(__name__)

def norm(s: str) -> str:
    """文字列正規化関数"""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    s = re.sub(r'[\s　・,，\.。／/()-]', '', s)
    return s

def ratio(a: str, b: str) -> float:
    """2つの文字列の類似度を計算"""
    return SequenceMatcher(None, a, b).ratio()

def common_memo(m: str) -> str:
    """摘要から共通部分を抽出（バランス検査用）"""
    return m.split("; 借方:",1)[0].split("; 貸方:",1)[0].strip()

class MJSConverter:
    """ミロク取込45列CSV変換器クラス"""
    
    # ミロク取込45列ヘッダー（完全一致）
    MJS_45_COLUMNS = [
        "伝票日付", "内部月", "伝票NO", "証憑NO", "データ種別", "仕訳入力形式",
        "（借）科目ｺｰﾄﾞ", "（借）補助ｺｰﾄﾞ", "（借）部門ｺｰﾄﾞ", "（借）セグメントｺｰﾄﾞ",
        "（借）消費税区分", "（借）業種", "（借）税込区分", "（借）補助区分1",
        "（借）補助ｺｰﾄﾞ1", "（借）補助区分2", "（借）補助ｺｰﾄﾞ2",
        "（貸）科目ｺｰﾄﾞ", "（貸）補助ｺｰﾄﾞ", "（貸）部門ｺｰﾄﾞ", "（貸）セグメントｺｰﾄﾞ",
        "（貸）消費税区分", "（貸）業種", "（貸）税込区分", "（貸）補助区分1",
        "（貸）補助ｺｰﾄﾞ1", "（貸）補助区分2", "（貸）補助ｺｰﾄﾞ2",
        "金額", "消費税額", "消費税ｺｰﾄﾞ", "消費税率", "外税同時入力区分",
        "資金繰入力区分", "資金繰ｺｰﾄﾞ", "摘要", "摘要コード1", "摘要コード2",
        "摘要コード3", "摘要コード4", "摘要コード5", "期日", "付箋", "付箋コメント",
        "事業者取引区分"
    ]
    
    def __init__(self):
        """初期化"""
        self.account_codes = {}  # {正規化科目名: (科目コード, 補助コード)}
        logger.info("MJSConverter initialized")
    
    def load_account_codes(self, account_code_csv_path: str) -> None:
        """勘定科目コード一覧CSVを読み込み"""
        try:
            # Shift-JIS → UTF-8-sig フォールバック読み込み
            try:
                df = pd.read_csv(account_code_csv_path, encoding='shift_jis')
                logger.info(f"Account code CSV loaded (shift_jis): {len(df)} records")
            except UnicodeDecodeError:
                logger.warning("Shift-JIS decode failed, trying UTF-8-sig...")
                df = pd.read_csv(account_code_csv_path, encoding='utf-8-sig')
                logger.info(f"Account code CSV loaded (utf-8-sig): {len(df)} records")
            
            # カラム名の候補を定義（半角カタカナ含む）
            code_candidates = ["科目ｺｰﾄﾞ", "科目コード", "勘定科目コード", "コード"]
            name_candidates = ["科目名", "勘定科目名", "名称"]
            aux_code_candidates = ["補助ｺｰﾄﾞ", "補助コード", "補助科目コード"]
            aux_name_candidates = ["補助科目名", "補助名称"]
            
            # 実際のカラム名を特定
            code_col = self._find_column(df.columns, code_candidates)
            name_col = self._find_column(df.columns, name_candidates)
            aux_code_col = self._find_column(df.columns, aux_code_candidates)
            aux_name_col = self._find_column(df.columns, aux_name_candidates)
            
            if not code_col or not name_col:
                raise ValueError(f"Required columns not found. Available: {list(df.columns)}")
            
            logger.info(f"Using columns: code={code_col}, name={name_col}, aux_code={aux_code_col}")
            
            # 辞書に格納
            for _, row in df.iterrows():
                account_name = str(row[name_col]).strip()
                account_code = str(row[code_col]).strip()
                
                aux_code = ""
                if aux_code_col and pd.notna(row[aux_code_col]):
                    aux_code = str(row[aux_code_col]).strip()
                
                if account_name and account_code:
                    normalized_name = norm(account_name)
                    self.account_codes[normalized_name] = (account_code, aux_code)
                    
                    # 補助科目名も追加（もしあれば）
                    if aux_name_col and pd.notna(row[aux_name_col]):
                        aux_name = str(row[aux_name_col]).strip()
                        if aux_name:
                            normalized_aux_name = norm(aux_name)
                            self.account_codes[normalized_aux_name] = (account_code, aux_code)
            
            logger.info(f"Account codes loaded: {len(self.account_codes)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load account codes: {e}")
            raise
    
    def _find_column(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        """カラム名の候補から実際のカラム名を見つける"""
        for candidate in candidates:
            for col in columns:
                if candidate in col or col in candidate:
                    return col
        return None
    
    def lookup_account_code(self, account_name: str) -> Tuple[str, str]:
        """科目名から科目コードと補助コードを検索"""
        if not account_name.strip():
            return "", ""
        
        normalized_name = norm(account_name)
        logger.debug(f"Looking up account: '{account_name}' -> normalized: '{normalized_name}'")
        
        # 1. 完全一致
        if normalized_name in self.account_codes:
            result = self.account_codes[normalized_name]
            logger.debug(f"Exact match found: {result}")
            return result
        
        # 2. 前方/部分一致
        matches = []
        for stored_name, codes in self.account_codes.items():
            if normalized_name in stored_name or stored_name in normalized_name:
                matches.append((stored_name, codes))
                logger.debug(f"Partial match: '{stored_name}' -> {codes}")
        
        # ユニークに決まる場合のみ
        if len(matches) == 1:
            result = matches[0][1]
            logger.debug(f"Unique partial match found: {result}")
            return result
        elif len(matches) > 1:
            logger.debug(f"Multiple partial matches found ({len(matches)}), skipping")
        
        # 3. 近似一致（しきい値 0.85）
        best_ratio = 0.0
        best_match = None
        best_stored_name = ""
        
        for stored_name, codes in self.account_codes.items():
            r = ratio(normalized_name, stored_name)
            if r > best_ratio and r >= 0.85:
                best_ratio = r
                best_match = codes
                best_stored_name = stored_name
        
        if best_match:
            logger.debug(f"Fuzzy match found: '{best_stored_name}' (ratio: {best_ratio:.3f}) -> {best_match}")
            return best_match
        
        # 見つからない場合は空文字
        logger.debug(f"No match found for '{account_name}'")
        return "", ""
    
    def normalize_date(self, date_str: str) -> str:
        """日付をYYYY/M/D形式に正規化（和暦→西暦）"""
        if not date_str or not date_str.strip():
            return ""
        
        date_str = date_str.strip()
        
        # 和暦の変換（令和）
        if date_str.startswith('R') or date_str.startswith('r'):
            try:
                # R07/02/12 -> 2025/2/12
                parts = re.findall(r'\d+', date_str)
                if len(parts) >= 3:
                    reiwa_year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    western_year = 2018 + reiwa_year  # 令和元年は2019年
                    return f"{western_year}/{month}/{day}"
            except (ValueError, IndexError):
                pass
        
        # すでに西暦の場合はそのまま（YYYY/M/D形式に調整）
        try:
            if '/' in date_str:
                parts = date_str.split('/')
                if len(parts) >= 3:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    return f"{year}/{month}/{day}"
        except (ValueError, IndexError):
            pass
        
        # 変換できない場合はそのまま返す
        return date_str
    
    def normalize_amount(self, amount_str: str) -> int:
        """金額を正の整数に正規化"""
        if not amount_str:
            return 0
        
        try:
            # カンマや通貨記号を除去
            cleaned = re.sub(r'[,，￥¥$]', '', str(amount_str))
            amount = float(cleaned)
            
            # 負値は0にする
            if amount < 0:
                return 0
            
            return int(amount)
        
        except (ValueError, TypeError):
            return 0
    
    def create_mjs_row(self, entry: Dict[str, Any]) -> Dict[str, str]:
        """5カラムJSONエントリから45列の行を作成"""
        # 45列の空行を作成
        row = {col: "" for col in self.MJS_45_COLUMNS}
        
        # 基本情報の設定
        row["伝票日付"] = self.normalize_date(entry.get("伝票日付", ""))
        row["金額"] = str(self.normalize_amount(entry.get("金額", 0)))
        row["消費税額"] = "0"  # 固定値
        row["摘要"] = str(entry.get("摘要", ""))
        
        # 科目コードの補完
        account_name = str(entry.get("科目名", "")).strip()
        account_code, aux_code = self.lookup_account_code(account_name)
        
        # 借貸区分に応じてコードを設定
        debit_credit = str(entry.get("借貸区分", "")).strip()
        
        if debit_credit == "借方":
            row["（借）科目ｺｰﾄﾞ"] = account_code
            row["（借）補助ｺｰﾄﾞ"] = aux_code
        elif debit_credit == "貸方":
            row["（貸）科目ｺｰﾄﾞ"] = account_code
            row["（貸）補助ｺｰﾄﾞ"] = aux_code
        
        return row
    
    def validate_and_warn(self, data: List[Dict[str, Any]]) -> None:
        """軽量バリデーションと警告ログ"""
        logger.info(f"Validating {len(data)} entries...")
        
        # 基本バリデーション
        for i, row in enumerate(data):
            # 借方系と貸方系のどちらか一方のみコード欄が埋まっているか
            has_debit = bool(row.get("（借）科目ｺｰﾄﾞ", ""))
            has_credit = bool(row.get("（貸）科目ｺｰﾄﾞ", ""))
            
            if has_debit and has_credit:
                logger.warning(f"Row {i}: Both debit and credit codes are set")
            elif not has_debit and not has_credit:
                logger.warning(f"Row {i}: Neither debit nor credit codes are set")
            
            # 金額が正の数か
            try:
                amount = float(row.get("金額", "0"))
                if amount < 0:
                    logger.warning(f"Row {i}: Negative amount: {amount}")
            except (ValueError, TypeError):
                logger.warning(f"Row {i}: Invalid amount format: {row.get('金額')}")
        
        # 簡易グルーピングとバランス検査
        self._check_balance(data)
    
    def _check_balance(self, data: List[Dict[str, Any]]) -> None:
        """簡易グルーピングで借方合計=貸方合計を確認"""
        # 同日付かつ摘要の共通部分が一致するものをグループ化
        groups = {}
        
        for row in data:
            date = row.get("伝票日付", "")
            memo = row.get("摘要", "")
            common_part = common_memo(memo)
            
            key = (date, common_part)
            if key not in groups:
                groups[key] = {"debit_total": 0, "credit_total": 0}
            
            amount_str = row.get("金額", "0")
            try:
                amount = float(amount_str)
            except (ValueError, TypeError):
                continue
            
            if row.get("（借）科目ｺｰﾄﾞ"):
                groups[key]["debit_total"] += amount
            elif row.get("（貸）科目ｺｰﾄﾞ"):
                groups[key]["credit_total"] += amount
        
        # バランスチェック
        unbalanced_count = 0
        for (date, memo), totals in groups.items():
            debit = totals["debit_total"]
            credit = totals["credit_total"]
            
            if abs(debit - credit) > 0.01:  # 1円未満の誤差は許容
                unbalanced_count += 1
                logger.warning(f"Balance mismatch: {date} '{memo[:50]}...' - "
                             f"Debit: {debit}, Credit: {credit}, Diff: {debit - credit}")
        
        if unbalanced_count == 0:
            logger.info("All groups are balanced")
        else:
            logger.warning(f"{unbalanced_count} unbalanced groups found")


def fivejson_to_mjs45(
    ai_json_path: str,
    account_code_csv_path: str,
    out_csv_path: str,
    log_path: Optional[str] = None
) -> None:
    """
    5カラムJSONをミロク取込45列CSVに変換し、科目名からコードを補完する。
    
    Args:
        ai_json_path: 5カラムJSONファイルのパス
        account_code_csv_path: 勘定科目コード一覧CSVのパス
        out_csv_path: 出力CSVファイルのパス
        log_path: ログファイルのパス（オプション）
    """
    # ログ設定
    if log_path:
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    try:
        logger.info(f"Starting conversion: {ai_json_path} -> {out_csv_path}")
        
        # 1. JSONデータの読み込み
        logger.info(f"Loading JSON from: {ai_json_path}")
        with open(ai_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if not isinstance(json_data, list):
            raise ValueError("JSON data must be an array")
        
        logger.info(f"Loaded {len(json_data)} entries from JSON")
        
        # 2. データの正規化
        normalized_data = []
        for i, entry in enumerate(json_data):
            if not isinstance(entry, dict):
                logger.warning(f"Entry {i} is not a dictionary, skipping")
                continue
            
            # 必須フィールドの確認
            required_fields = ["伝票日付", "借貸区分", "科目名", "金額", "摘要"]
            missing_fields = [field for field in required_fields if field not in entry]
            
            if missing_fields:
                logger.warning(f"Entry {i} missing fields: {missing_fields}")
                # 欠損フィールドに空値を設定
                for field in missing_fields:
                    entry[field] = "" if field != "金額" else 0
            
            # NFKC正規化
            normalized_entry = {}
            for key, value in entry.items():
                normalized_key = unicodedata.normalize("NFKC", str(key))
                normalized_value = unicodedata.normalize("NFKC", str(value))
                normalized_entry[normalized_key] = normalized_value
            
            normalized_data.append(normalized_entry)
        
        logger.info(f"Normalized {len(normalized_data)} entries")
        
        # 3. MJSConverterの初期化と勘定科目コードの読み込み
        converter = MJSConverter()
        converter.load_account_codes(account_code_csv_path)
        
        # 4. 45列データの生成
        mjs_data = []
        for entry in normalized_data:
            mjs_row = converter.create_mjs_row(entry)
            mjs_data.append(mjs_row)
        
        logger.info(f"Generated {len(mjs_data)} MJS rows")
        
        # 5. バリデーション
        converter.validate_and_warn(mjs_data)
        
        # 6. CSVファイルの保存
        logger.info(f"Saving CSV to: {out_csv_path}")
        
        # 出力ディレクトリの作成
        Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(
                csvfile, 
                fieldnames=MJSConverter.MJS_45_COLUMNS,
                quoting=csv.QUOTE_MINIMAL
            )
            writer.writeheader()
            writer.writerows(mjs_data)
        
        logger.info(f"Successfully saved {len(mjs_data)} rows to {out_csv_path}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python mjs_converter.py <ai_json_path> <account_code_csv_path> <out_csv_path> [log_path]")
        sys.exit(1)
    
    ai_json_path = sys.argv[1]
    account_code_csv_path = sys.argv[2]
    out_csv_path = sys.argv[3]
    log_path = sys.argv[4] if len(sys.argv) > 4 else None
    
    # コンソールログの設定（デバッグレベル）
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        fivejson_to_mjs45(ai_json_path, account_code_csv_path, out_csv_path, log_path)
        print(f"Conversion completed successfully: {out_csv_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)