#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
後処理モジュール - 貸借ペア保証と金額バリデーション
LLM抽出後の5カラムJSON配列に対し、データ品質を保証する
"""

import re
import logging
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from utils.reconcile_dedupe import reconcile_and_dedupe

logger = logging.getLogger(__name__)

def _common_memo(text: str) -> str:
    """
    摘要から共通部分を抽出（「; 借方:」「; 貸方:」より前）
    
    Args:
        text: 摘要文字列
        
    Returns:
        共通摘要部分
    """
    if not text:
        return ""
    t = text.split("; 借方:", 1)[0].split("; 貸方:", 1)[0].strip()
    return t

def enforce_debit_credit_pairs(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    LLM抽出後の5カラムJSON配列に対し、貸借ペアを強制保証
    
    処理内容:
    - 同一枠（伝票日付＋共通摘要）でグルーピング
    - 借方/貸方のいずれかが欠けていれば補完レッグを自動生成
    - 摘要末尾の ; 借方: / 貸方: が無い・片側のみ → 不明側を '不明【OCR注意】' にして強制併記
    - 枠内の借貸合計を一致させる（不足差額分は"不明"レッグで補う）
    
    Args:
        entries: LLM抽出後の5カラムJSON配列
        
    Returns:
        貸借ペアが保証された5カラムJSON配列
    """
    if not entries:
        return []
    
    logger.info(f"Starting debit-credit pair enforcement for {len(entries)} entries")
    
    # 同一枠でグルーピング
    groups = defaultdict(list)
    for e in entries:
        key = (e.get("伝票日付", ""), _common_memo(e.get("摘要", "")))
        groups[key].append(e)
    
    fixed = []
    for (date, memo), rows in groups.items():
        # 借方・貸方の分類
        debit_rows = [r for r in rows if r.get("借貸区分") == "借方"]
        credit_rows = [r for r in rows if r.get("借貸区分") == "貸方"]
        
        # 既存行から相手側科目名を抽出
        def find_side_name(rows: List[Dict], side: str) -> str:
            """摘要から指定サイドの科目名を抽出"""
            pattern = r";\s*" + side + r":(.+?)(?:\s|$|/)"
            for r in rows:
                memo_text = r.get("摘要", "")
                match = re.search(pattern, memo_text)
                if match:
                    return match.group(1).strip()
            return ""
        
        debit_side_name = find_side_name(rows, "借方")
        credit_side_name = find_side_name(rows, "貸方")
        
        # 摘要末尾の正規化関数
        def normalize_memo(memo: str, d_name: str, c_name: str) -> str:
            """摘要を正規化し、必ず両サイド併記にする"""
            d = d_name if d_name else "不明【OCR注意】"
            c = c_name if c_name else "不明【OCR注意】"
            base = _common_memo(memo)
            return f"{base} ; 借方:{d} / 貸方:{c}".strip()
        
        # 片側欠落の補完
        if not debit_rows and rows:
            # 借方が存在しない場合、貸方の金額から補完
            any_row = rows[0]
            total_credit_amount = sum(int(r.get("金額", 0)) for r in credit_rows)
            
            supplement_row = {
                "伝票日付": any_row.get("伝票日付", ""),
                "借貸区分": "借方",
                "科目名": debit_side_name,  # 空なら後段で【科目コード要確認】が付く
                "金額": total_credit_amount,
                "摘要": normalize_memo(any_row.get("摘要", ""), debit_side_name, credit_side_name)
            }
            fixed.append(supplement_row)
            debit_rows = [supplement_row]
        
        if not credit_rows and rows:
            # 貸方が存在しない場合、借方の金額から補完
            any_row = rows[0]
            total_debit_amount = sum(int(r.get("金額", 0)) for r in debit_rows)
            
            supplement_row = {
                "伝票日付": any_row.get("伝票日付", ""),
                "借貸区分": "貸方",
                "科目名": credit_side_name,
                "金額": total_debit_amount,
                "摘要": normalize_memo(any_row.get("摘要", ""), debit_side_name, credit_side_name)
            }
            fixed.append(supplement_row)
            credit_rows = [supplement_row]
        
        # 金額整合性チェックと補完
        sum_debit = sum(int(r.get("金額", 0)) for r in debit_rows)
        sum_credit = sum(int(r.get("金額", 0)) for r in credit_rows)
        
        if sum_debit != sum_credit:
            diff = abs(sum_debit - sum_credit)
            any_row = rows[0] if rows else {}
            missing_side = "貸方" if sum_debit > sum_credit else "借方"
            
            # 差額補完レッグを追加
            balance_row = {
                "伝票日付": any_row.get("伝票日付", ""),
                "借貸区分": missing_side,
                "科目名": "",  # 空 → 後段で【科目コード要確認】が付く
                "金額": diff,
                "摘要": normalize_memo(any_row.get("摘要", ""), debit_side_name, credit_side_name) + " 【差額補完】"
            }
            fixed.append(balance_row)
            
            logger.warning(f"Amount imbalance fixed: {missing_side} +{diff} for memo '{memo[:30]}...'")
        
        # 既存行の摘要正規化
        for r in rows:
            r["摘要"] = normalize_memo(r.get("摘要", ""), debit_side_name, credit_side_name)
            fixed.append(r)
    
    logger.info(f"Debit-credit pair enforcement completed: {len(entries)} -> {len(fixed)} entries")
    return fixed

def validate_amounts(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    金額バリデーション - 金額=0の行をブロック & 可視化
    
    Args:
        entries: 5カラムJSON配列
        
    Returns:
        Tuple[valid_entries, error_entries]: (有効なエントリ, エラーエントリ)
    """
    if not entries:
        return [], []
    
    logger.info(f"Starting amount validation for {len(entries)} entries")
    
    valid_entries = []
    error_entries = []
    
    for e in entries:
        try:
            amount = int(e.get("金額", 0) or 0)
        except (ValueError, TypeError):
            amount = 0
        
        if amount <= 0:
            # 金額が0または負の場合はエラーエントリに分類
            error_entry = e.copy()
            error_entry["摘要"] = (error_entry.get("摘要", "") + " 【OCR注意:金額読取不可】").strip()
            error_entries.append(error_entry)
            logger.warning(f"Zero/negative amount detected: {error_entry.get('摘要', '')[:50]}...")
        else:
            valid_entries.append(e)
    
    logger.info(f"Amount validation completed: {len(valid_entries)} valid, {len(error_entries)} errors")
    return valid_entries, error_entries

def postprocess_extracted_data(entries: List[Dict[str, Any]], config=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    抽出データの総合後処理
    
    1. 貸借ペア保証
    2. 金額バリデーション
    3. 重複除去と残高調整（合算行除去・one-vs-many分割）
    
    Args:
        entries: LLM抽出後の生データ
        config: 設定オブジェクト（将来の設定拡張用）
        
    Returns:
        Tuple[processed_entries, error_entries]: (処理済みデータ, エラーデータ)
    """
    logger.info("Starting comprehensive post-processing")
    
    # 1. 貸借ペア保証
    paired_entries = enforce_debit_credit_pairs(entries)
    
    # 2. 金額バリデーション
    valid_entries, error_entries = validate_amounts(paired_entries)
    
    # 3. 重複除去と残高調整（合算行・分割処理）
    if valid_entries:
        valid_entries = reconcile_and_dedupe(valid_entries)
    
    logger.info(f"Post-processing completed: {len(valid_entries)} valid entries, {len(error_entries)} errors")
    
    return valid_entries, error_entries