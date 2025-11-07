#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仕訳No付番モジュール - 貸借一致ブロック方式
OCRから得られた伝票データに対して「1取引＝貸借一致ブロック＝1仕訳No」の原則に従い、
安定的な仕訳番号を自動付番する。

設計原則：
- 逐次走査＋貸借金額一致で1伝票を確定
- ヘッダ語句（更新、退去、契約者など）が出現したら強制ブロック化
- 誤差±1円までは一致とみなす
- 1ブロック50行超えたら強制クローズ
- LLMの出力仕訳Noは無視し、後段で一元管理
"""

import logging
import re
import uuid
import hashlib
from typing import List, Dict, Any, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)

# ヘッダ行判定用キーワード（正規表現）
HEADER_KEYWORDS = re.compile(
    r"(更新.{0,2}伝票|退去|契約.?者|物件.?名|号.?室|オーナー|預.?り.?金|敷.?金|解約|"
    r"新規|振替|駐車場|"
    r"伝票(番号|No\.?)|請求(書)?番号|受付番号|入金|出金|振込|送金)"
)

# 1ブロックの最大行数（安全弁）
MAX_BLOCK_SIZE = 50

# 早期フラッシュの推奨サイズ（パフォーマンス最適化）
RECOMMENDED_BLOCK_SIZE = 20

# 貸借一致の許容誤差（円）
BALANCE_TOLERANCE = 1

# 累積金額の上限（異常に大きなブロックを防ぐ）
MAX_CUMULATIVE_AMOUNT = 100_000_000  # 1億円


def is_header_line(row: Dict[str, Any]) -> bool:
    """
    ヘッダ行かどうかを判定する

    Args:
        row: 仕訳データ行

    Returns:
        bool: ヘッダ行の場合True
    """
    memo = row.get("摘要", "")
    account = row.get("科目名", "")

    # 摘要または科目名にヘッダキーワードが含まれる場合
    return bool(HEADER_KEYWORDS.search(memo)) or bool(HEADER_KEYWORDS.search(account))


def generate_block_uuid(rows: List[Dict[str, Any]]) -> str:
    """
    ブロックのUUIDv5ハッシュを生成（監査用）

    Args:
        rows: ブロック内の仕訳データ行

    Returns:
        str: UUIDv5ハッシュ
    """
    if not rows:
        return ""

    # ブロックの特徴的な情報を結合してハッシュ化
    block_signature = ""
    for row in rows:
        block_signature += f"{row.get('伝票日付', '')}"
        block_signature += f"{row.get('借貸区分', '')}"
        block_signature += f"{row.get('科目名', '')}"
        block_signature += f"{row.get('金額', 0)}"
        block_signature += f"{row.get('摘要', '')[:50]}"  # 摘要は最初の50文字のみ

    # UUIDv5を生成（namespace: URL）
    namespace = uuid.NAMESPACE_URL
    return str(uuid.uuid5(namespace, block_signature))


def assign_voucher_numbers(
    rows: List[Dict[str, Any]],
    prefer_llm: bool = False,  # デフォルトFalseに変更（新ロジックを優先）
    width: int = 4  # 4桁にデフォルト変更（0001〜9999）
) -> List[Dict[str, Any]]:
    """
    貸借一致ブロック方式で仕訳No付番を行う

    Args:
        rows: LLM出力済みJSON（list[dict]）
        prefer_llm: 旧バージョン互換用（無視される）
        width: 連番のゼロ埋め桁数（デフォルト4）

    Returns:
        仕訳No、block_uuid、warning、差額が付与されたJSON配列
    """
    if not rows:
        logger.warning("assign_voucher_numbers: 入力が空です")
        return []

    logger.info(f"仕訳No付番開始（貸借一致ブロック方式）: {len(rows)}行, width={width}")

    # 処理結果を格納するリスト
    result = []

    # 現在のブロック
    current_block = []
    sum_debit = 0  # 借方合計
    sum_credit = 0  # 貸方合計
    voucher_seq = 0  # 仕訳No連番

    # 統計情報
    stats = {
        "total_blocks": 0,
        "balanced_blocks": 0,
        "header_closes": 0,
        "balance_closes": 0,
        "early_flush_closes": 0,
        "amount_overflow_closes": 0,
        "overflow_closes": 0,
        "eof_closes": 0,
        "unbalanced_blocks": 0
    }

    def flush_block(reason: str) -> None:
        """現在のブロックを確定して結果に追加する"""
        nonlocal current_block, sum_debit, sum_credit, voucher_seq, stats

        if not current_block:
            return

        # 借方・貸方がともに1件以上存在する場合のみ採番
        has_debit = any(r.get("借貸区分") == "借方" for r in current_block)
        has_credit = any(r.get("借貸区分") == "貸方" for r in current_block)

        if not (has_debit and has_credit):
            logger.warning(f"ブロック不完全（借方または貸方が欠落）: {len(current_block)}行をスキップ - reason={reason}")
            current_block.clear()
            sum_debit = sum_credit = 0
            return

        # 貸借差額を計算
        balance_diff = sum_debit - sum_credit
        is_balanced = abs(balance_diff) <= BALANCE_TOLERANCE

        # 仕訳No採番
        voucher_seq += 1
        voucher_no = str(voucher_seq).zfill(width)

        # UUIDハッシュ生成
        block_uuid = generate_block_uuid(current_block)

        # 警告メッセージ生成
        warnings = []
        if not is_balanced:
            warnings.append(f"UNBALANCED(借方={sum_debit}, 貸方={sum_credit}, 差額={balance_diff})")
            stats["unbalanced_blocks"] += 1
        if reason == "OVERFLOW":
            warnings.append(f"OVERFLOW({len(current_block)}行)")
        if reason == "HEADER_BOUNDARY":
            stats["header_closes"] += 1
        elif reason == "BALANCE_CLOSE":
            stats["balance_closes"] += 1
        elif reason == "EARLY_FLUSH":
            stats["early_flush_closes"] += 1
        elif reason == "AMOUNT_OVERFLOW":
            stats["amount_overflow_closes"] += 1
        elif reason == "OVERFLOW":
            stats["overflow_closes"] += 1
        elif reason == "EOF":
            stats["eof_closes"] += 1

        warning_str = " | ".join(warnings) if warnings else ""

        # ブロック内の全行に情報を付与
        for row in current_block:
            row["仕訳No"] = voucher_no
            row["block_uuid"] = block_uuid
            row["warning"] = warning_str
            row["差額"] = balance_diff
            result.append(row)

        # 統計更新
        stats["total_blocks"] += 1
        if is_balanced:
            stats["balanced_blocks"] += 1

        logger.debug(
            f"仕訳No={voucher_no}: {len(current_block)}行, "
            f"借方={sum_debit}, 貸方={sum_credit}, 差額={balance_diff}, "
            f"reason={reason}, warnings={warning_str}"
        )

        # ブロックをクリア
        current_block.clear()
        sum_debit = sum_credit = 0

    # 逐次走査でブロック化
    for idx, row in enumerate(rows):
        # ヘッダ行検出 → 現在のブロックをクローズして新ブロック開始
        if is_header_line(row) and current_block:
            flush_block("HEADER_BOUNDARY")

        # (オプション) 同一摘要連続性チェック: 最初の行と摘要カテゴリが大きく異なる場合はクローズ
        # 注: 厳しすぎる場合はこの条件を無効化できます
        # if current_block and len(current_block) > 0:
        #     first_memo = current_block[0].get("摘要", "")[:20]
        #     current_memo = row.get("摘要", "")[:20]
        #     if first_memo != current_memo and len(current_block) >= 2:
        #         flush_block("MEMO_CHANGE")

        # 現在のブロックに行を追加
        current_block.append(row)

        # 金額を累積
        side = row.get("借貸区分", "")
        amount = row.get("金額", 0)
        try:
            amount = float(amount) if amount else 0
        except (ValueError, TypeError):
            amount = 0

        if side == "借方":
            sum_debit += amount
        elif side == "貸方":
            sum_credit += amount

        # 貸借一致判定 → ブロックをクローズ
        # 条件: 貸借一致 AND 同一日付
        is_balanced = (
            sum_debit > 0
            and sum_credit > 0
            and abs(sum_debit - sum_credit) <= BALANCE_TOLERANCE
            and len({r.get("伝票日付", "") for r in current_block}) == 1
        )

        if is_balanced:
            flush_block("BALANCE_CLOSE")

        # 早期フラッシュ判定（パフォーマンス最適化）
        # 推奨サイズに達した場合、貸借がほぼ一致していればフラッシュ
        elif (
            len(current_block) >= RECOMMENDED_BLOCK_SIZE
            and sum_debit > 0
            and sum_credit > 0
            and abs(sum_debit - sum_credit) <= BALANCE_TOLERANCE * 10  # 許容誤差を10倍に緩和
        ):
            flush_block("EARLY_FLUSH")
            logger.debug(f"早期フラッシュ: {len(current_block)}行, 差額={abs(sum_debit - sum_credit)}円")

        # 累積金額上限チェック → 異常に大きなブロックを防ぐ
        elif (sum_debit + sum_credit) > MAX_CUMULATIVE_AMOUNT:
            flush_block("AMOUNT_OVERFLOW")
            logger.warning(f"累積金額上限超過: 借方={sum_debit}円, 貸方={sum_credit}円")

        # オーバーフロー判定 → 強制クローズ
        elif len(current_block) >= MAX_BLOCK_SIZE:
            flush_block("OVERFLOW")

    # 最後のブロックをクローズ
    if current_block:
        flush_block("EOF")

    # 統計情報をログ出力
    logger.info(
        f"仕訳No付番完了: {len(result)}行, {stats['total_blocks']}ブロック, "
        f"一致={stats['balanced_blocks']}, 不一致={stats['unbalanced_blocks']}, "
        f"ヘッダ分割={stats['header_closes']}, 金額一致={stats['balance_closes']}, "
        f"早期フラッシュ={stats['early_flush_closes']}, 金額上限={stats['amount_overflow_closes']}, "
        f"オーバーフロー={stats['overflow_closes']}, EOF={stats['eof_closes']}"
    )

    return result


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


def validate_overall_balance(rows: List[Dict[str, Any]]) -> Tuple[bool, float, float]:
    """
    全仕訳の借方合計＝貸方合計を検証する

    Args:
        rows: 仕訳データ配列

    Returns:
        Tuple[is_balanced, total_debit, total_credit]
    """
    total_debit = 0
    total_credit = 0

    for row in rows:
        side = row.get("借貸区分", "")
        amount = row.get("金額", 0)
        try:
            amount = float(amount) if amount else 0
        except (ValueError, TypeError):
            amount = 0

        if side == "借方":
            total_debit += amount
        elif side == "貸方":
            total_credit += amount

    is_balanced = abs(total_debit - total_credit) <= BALANCE_TOLERANCE

    logger.info(
        f"全体バランス検証: 借方合計={total_debit:.2f}, 貸方合計={total_credit:.2f}, "
        f"差額={total_debit - total_credit:.2f}, 一致={is_balanced}"
    )

    return is_balanced, total_debit, total_credit
