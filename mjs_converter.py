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
        self.alias_map = {}  # {エイリアス: (科目コード, 補助コード) or 正規化科目名}
        self.account_name_map = {}  # 高速検索用: {正規化科目名: (科目コード, 補助コード)}
        logger.info("MJSConverter initialized")
    
    def load_account_codes(self, account_code_csv_path: str) -> None:
        """勘定科目コード一覧CSVを読み込み"""
        try:
            # ファイルの存在確認
            from pathlib import Path
            if not Path(account_code_csv_path).exists():
                logger.error(f"Account code CSV file not found: {account_code_csv_path}")
                raise FileNotFoundError(f"Account code CSV file not found: {account_code_csv_path}")
            
            # CP932 → UTF-8-sig 二段トライ（件数ログ強化）
            try:
                df = pd.read_csv(account_code_csv_path, encoding='cp932')
                logger.info(f"Account code CSV loaded (cp932): {len(df)} records")
                encoding_used = "cp932"
            except UnicodeDecodeError:
                logger.warning("CP932 decode failed, trying UTF-8-sig...")
                try:
                    df = pd.read_csv(account_code_csv_path, encoding='utf-8-sig')
                    logger.info(f"Account code CSV loaded (utf-8-sig): {len(df)} records")
                    encoding_used = "utf-8-sig"
                except Exception as e:
                    logger.error(f"Both encodings failed: {e}")
                    logger.warning("名寄せマスタ未適用")
                    raise
            
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
                    codes = (account_code, aux_code)
                    self.account_codes[normalized_name] = codes
                    self.account_name_map[normalized_name] = codes
                    
                    # 補助科目名も追加（もしあれば）
                    if aux_name_col and pd.notna(row[aux_name_col]):
                        aux_name = str(row[aux_name_col]).strip()
                        if aux_name:
                            normalized_aux_name = norm(aux_name)
                            self.account_codes[normalized_aux_name] = codes
                            self.account_name_map[normalized_aux_name] = codes
            
            # エイリアスCSVの読み込み
            self.load_aliases()
            
            # 主要科目の存在チェック（INFO出力）
            key_accounts = ["普通預金", "当座", "預り金", "敷金", "売上高", "営繕費", "修繕費"]
            found_accounts = []
            for key_account in key_accounts:
                if any(key_account in name for name in self.account_codes.keys()):
                    found_accounts.append(key_account)
            
            logger.info(f"Account codes loaded: {len(self.account_codes)} entries, encoding: {encoding_used}")
            logger.info(f"Key accounts found: {found_accounts}")
            logger.info(f"Aliases loaded: {len(self.alias_map)} entries")
            
            # マスタ読み込み成功フラグ
            self._master_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load account codes: {e}")
            logger.warning("名寄せマスタ未適用")
            # マスタ読み込み失敗フラグ
            self._master_loaded = False
            # 失敗時は空の辞書で続行（全エントリに未適用タグが付く）
            self.account_codes = {}
            self.alias_map = {}
            self.account_name_map = {}
    
    def _find_column(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        """カラム名の候補から実際のカラム名を見つける"""
        for candidate in candidates:
            for col in columns:
                if candidate in col or col in candidate:
                    return col
        return None
    
    def load_aliases(self) -> None:
        """エイリアスCSVを読み込み（存在する場合）"""
        alias_csv_path = Path("科目エイリアス.csv")
        if not alias_csv_path.exists():
            logger.info("Alias CSV not found, skipping alias loading")
            return
        
        try:
            import csv
            with alias_csv_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    alias = self._norm_alias(row.get("alias", ""))
                    if not alias:
                        continue
                    
                    # 直接コード指定の場合
                    if row.get("code"):
                        self.alias_map[alias] = (
                            str(row["code"]).strip(), 
                            str(row.get("aux_code", "")).strip()
                        )
                    # 正規化科目名への参照の場合
                    elif row.get("target_name"):
                        self.alias_map[alias] = str(row["target_name"]).strip()
            
            logger.info(f"Loaded {len(self.alias_map)} aliases from CSV")
        
        except Exception as e:
            logger.warning(f"Failed to load aliases: {e}")
    
    def _norm_alias(self, text: str) -> str:
        """エイリアス用の正規化（normよりも厳格）"""
        import unicodedata, re
        s = unicodedata.normalize("NFKC", text or "").strip()
        s = re.sub(r"\s+", "", s)
        return s
    
    def normalize_alias_name(self, name: str) -> str:
        """銀行ニックネームの正規化（表記ゆれ対応）"""
        n = self._norm_alias(name)
        
        # 表記ゆれのルール（運用に合わせて調整）
        rules = [
            (r"(西京|ｻｲｷｮｳ).*(普通|普)", "普通預金（西京）"),
            (r"(三菱|MUFG|UFJ).*(普通|普)", "普通預金（三菱）"),
            (r"(営繕|えいぜん)", "営繕費"),
            (r"(退居|退去).*(営繕|えいぜん)", "営繕費"),
        ]
        
        for pattern, canonical in rules:
            if re.search(pattern, n, flags=re.I):
                return canonical
        
        return name
    
    def lookup_account_code(self, account_name: str) -> Tuple[str, str]:
        """科目名から科目コードと補助コードを検索（エイリアス・ファジー対応強化）"""
        if not account_name or not account_name.strip():
            return "", ""
        
        # 1. エイリアス正規化
        normalized_name = self.normalize_alias_name(account_name)
        norm_name = norm(normalized_name)
        
        logger.debug(f"Looking up account: '{account_name}' -> normalized: '{normalized_name}' -> norm: '{norm_name}'")
        
        # 2. エイリアス直参照
        alias_norm = self._norm_alias(account_name)
        if alias_norm in self.alias_map:
            mapped = self.alias_map[alias_norm]
            if isinstance(mapped, tuple):   # (code, aux)
                logger.debug(f"Alias direct match: {mapped}")
                return mapped
            # target_name文字列なら、その名前で再検索
            logger.debug(f"Alias indirect match to: {mapped}")
            return self.lookup_account_code(mapped)
        
        # 3. 完全一致
        if norm_name in self.account_name_map:
            result = self.account_name_map[norm_name]
            logger.debug(f"Exact match found: {result}")
            return result
        
        # 4. 部分一致（最長一致＋語彙優先）
        matches = []
        PRIORITY = ["普通預金", "当座預金", "売上", "預り金", "立替金", "受取手数料", "敷金", "営繕", "保証金"]
        
        for stored_name, codes in self.account_name_map.items():
            if norm_name in stored_name or stored_name in norm_name:
                # スコア: (語彙ヒット数, 共通文字数, 候補名長の逆数)
                vocab_hits = sum(1 for keyword in PRIORITY if keyword in stored_name)
                common_chars = len(set(stored_name) & set(norm_name))
                length_score = 1.0 / max(len(stored_name), 1)
                
                score = (vocab_hits, common_chars, length_score)
                matches.append((score, stored_name, codes))
                logger.debug(f"Partial match: '{stored_name}' (score: {score}) -> {codes}")
        
        if matches:
            matches.sort(reverse=True)
            result = matches[0][2]
            logger.debug(f"Best partial match: {matches[0][1]} -> {result}")
            return result
        
        # 5. ファジー（閾値緩和: 0.78）
        best_ratio = 0.0
        best_match = None
        best_stored_name = ""
        
        for stored_name, codes in self.account_name_map.items():
            r = ratio(norm_name, stored_name)
            if r > best_ratio and r >= 0.78:  # 閾値緩和
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
    
    def _extract_side_from_memo(self, memo: str, side: str) -> str:
        """摘要からサイド別の科目名を抽出"""
        # side = "借方" or "貸方"
        m = re.search(rf"{side}:([^\s/]+)", memo or "")
        return m.group(1).strip() if m else ""

    def _decide_account_name(self, entry: Dict[str, Any]) -> str:
        """科目名の決定ロジック（空なら摘要からサイド別に拾う）"""
        # もともとの科目名
        name = (entry.get("科目名") or "").strip()
        if name:
            return name
        # 空なら摘要からサイド別に拾う
        side = entry.get("借貸区分", "")
        if side == "借方":
            return self._extract_side_from_memo(entry.get("摘要", ""), "借方")
        if side == "貸方":
            return self._extract_side_from_memo(entry.get("摘要", ""), "貸方")
        return ""  # 念のため

    def normalize_alias(self, name: str) -> str:
        """表記ゆれの正規化（最小セット）"""
        ALIAS = {
            "西京普通": "普通預金Ｂ",
            "三菱普通": "普通預金Ｃ",
            "売上": "売上高",
            "退居営繕費": "修繕費",
            "退去営繕費": "修繕費",
            "預り敷金": "敷金",
            "退居敷金": "敷金",
        }
        n = (name or "").strip()
        return ALIAS.get(n, n)

    def create_mjs_row(self, entry: Dict[str, Any]) -> Dict[str, str]:
        """5カラムJSONエントリから45列の行を作成"""
        # 45列の空行を作成
        row = {col: "" for col in self.MJS_45_COLUMNS}
        
        # 基本情報の設定
        row["伝票日付"] = self.normalize_date(entry.get("伝票日付", ""))
        row["金額"] = str(self.normalize_amount(entry.get("金額", 0)))
        row["消費税額"] = "0"  # 固定値
        row["摘要"] = str(entry.get("摘要", ""))
        
        # 科目名の決定と正規化
        account_name = self._decide_account_name(entry)  # A: 摘要からサイド別に拾う
        account_name = self.normalize_alias(account_name)  # B: エイリアスで名寄せ
        account_code, aux_code = self.lookup_account_code(account_name)
        
        # 未割当の場合は摘要に【科目コード要確認】を追記
        if not account_code and account_name:
            row["摘要"] = (row["摘要"] + " 【科目コード要確認】").strip()
        
        # マスタ読み込み失敗時の対応（全エントリに "名寄せマスタ未適用" タグ）
        if not hasattr(self, '_master_loaded') or not self._master_loaded:
            if not " 名寄せマスタ未適用" in row["摘要"]:
                row["摘要"] = (row["摘要"] + " 名寄せマスタ未適用").strip()
        
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

def save_or_mark_unconfirmed(output_csv_path: str, rows: list, source_basename: str):
    """CSVを保存するか、0行の場合は未確定CSVを別出力"""
    from pathlib import Path
    import csv

    if rows:
        # 正常な保存処理
        _save_csv(output_csv_path, rows)
        logger.info(f"Successfully saved {len(rows)} rows to {output_csv_path}")
    else:
        # 未確定ファイルを別出力
        unconfirmed_dir = Path("output/uncertain")
        unconfirmed_dir.mkdir(parents=True, exist_ok=True)
        unconfirmed_csv = unconfirmed_dir / f"{source_basename}_unconfirmed.csv"

        # 最低限のメッセージ付きCSVを出力
        fallback_row = {"メッセージ": "抽出失敗 or コード割当不可。Alias/マスタ/抽出設定を確認してください。"}
        _save_csv(unconfirmed_csv, [fallback_row])
        logger.warning(f"Zero rows for {output_csv_path}. Wrote placeholder to {unconfirmed_csv}")

def _save_csv(csv_path: str, rows: list):
    """CSVファイルを保存するヘルパー関数"""
    from pathlib import Path
    import csv

    # 出力ディレクトリの作成
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    # ヘッダーの決定
    if isinstance(rows[0], dict):
        # MJS 45列CSVの場合とそれ以外を判別
        first_row_keys = set(rows[0].keys())
        mjs_columns_set = set(MJSConverter.MJS_45_COLUMNS)

        if mjs_columns_set.issubset(first_row_keys) or "（借）科目ｺｰﾄﾞ" in first_row_keys:
            # MJS 45列CSVの場合
            fieldnames = MJSConverter.MJS_45_COLUMNS
        else:
            # 通常のCSV（メッセージ等）
            fieldnames = rows[0].keys()

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(rows)
    else:
        # リストの場合はそのまま書き込み
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(rows)

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
        
        # ファイル存在確認
        from pathlib import Path
        if not Path(ai_json_path).exists():
            logger.error(f"Input JSON file not found: {ai_json_path}")
            raise FileNotFoundError(f"Input JSON file not found: {ai_json_path}")
        
        if not Path(account_code_csv_path).exists():
            logger.error(f"Account code CSV file not found: {account_code_csv_path}")
            raise FileNotFoundError(f"Account code CSV file not found: {account_code_csv_path}")
        
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
        
        # 5.5 両コード空の行を除去（POSTPROCESS設定に応じて）
        from config import config
        if hasattr(config, 'POSTPROCESS') and config.POSTPROCESS.DROP_BOTH_CODE_EMPTY:
            rows_before = len(mjs_data)
            mjs_data = [row for row in mjs_data if not (
                row.get("（借）科目ｺｰﾄﾞ", "") == "" and row.get("（貸）科目ｺｰﾄﾞ", "") == ""
            )]
            rows_dropped = rows_before - len(mjs_data)
            if rows_dropped > 0:
                logger.info(f"Dedup/BothEmpty: dropped={rows_dropped} rows with both debit and credit codes empty")
        
        # 6. CSVファイルの保存（0行時は未確定CSV出力）
        logger.info(f"Saving CSV to: {out_csv_path}")

        # 入力ファイル名から拡張子を除いたベース名を取得
        source_basename = Path(ai_json_path).stem

        # 0行の場合は未確定CSV、通常の場合は正常保存
        save_or_mark_unconfirmed(out_csv_path, mjs_data, source_basename)
        
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