#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仕訳No付番モジュール
LLM出力済みJSON（list[dict]）に対して仕訳Noを付与・正規化する
"""

import logging
from typing import List, Dict, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)


def assign_voucher_numbers(rows: List[Dict[str, Any]], prefer_llm: bool = True, width: int = 3) -> List[Dict[str, Any]]:
    """
    仕訳No付番を一元化する関数

    Args:
        rows: LLM出力済みJSON（list[dict]）
        prefer_llm: TrueならLLMが出した仕訳Noを優先
        width: 連番のゼロ埋め桁数（デフォルト3）

    Returns:
        仕訳Noが付与されたJSON配列
    """
    if not rows:
        logger.warning("assign_voucher_numbers: 入力が空です")
        return []

    logger.info(f"仕訳No付番開始: {len(rows)}行, prefer_llm={prefer_llm}, width={width}")

    assigned = []
    current_no = 1

    # 1. グループ化キーの決定
    groups = OrderedDict()

    for row in rows:
        if prefer_llm and row.get("仕訳No"):
            # LLMが出した仕訳Noを優先
            key = row["仕訳No"]
        else:
            # 同一日付＋摘要でグループ化（枠単位）
            # 摘要から共通部分を抽出（ ; 借方: / ; 貸方: より前）
            memo = row.get("摘要", "")
            # セミコロン区切りの最初の部分を取得（共通摘要）
            common_memo = memo.split(";")[0].strip() if memo else ""
            key = (row.get("伝票日付", ""), common_memo)

        # グループに追加
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    # 2. グループごとに連番を付与
    for key, group_rows in groups.items():
        # 連番を生成（ゼロ埋め）
        voucher_no = str(current_no).zfill(width)

        # グループ内の全行に同じ仕訳Noを付与
        for row in group_rows:
            row["仕訳No"] = voucher_no
            assigned.append(row)

        logger.debug(f"仕訳No={voucher_no}: {len(group_rows)}行 (key={key})")
        current_no += 1

    logger.info(f"仕訳No付番完了: {len(assigned)}行, {len(groups)}グループ")
    return assigned


def sort_by_voucher_number(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    仕訳No順に並び替える（借方→貸方の順も保証）

    Args:
        rows: 仕訳Noが付与されたJSON配列

    Returns:
        並び替えられたJSON配列
    """
    if not rows:
        return []

    logger.info(f"仕訳No順ソート開始: {len(rows)}行")

    # ソートキー: 仕訳No → 借貸区分（借方=0, 貸方=1） → 科目名
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r.get("仕訳No", ""),
            0 if r.get("借貸区分") == "借方" else 1,
            r.get("科目名", "")
        )
    )

    logger.info(f"仕訳No順ソート完了: {len(sorted_rows)}行")
    return sorted_rows
