#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON抽出ガード機能
LLM生応答から安全にJSON配列を抽出・パースする
"""

import json
import re
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

REQUIRED_5_COLUMNS = ["伝票日付", "借貸区分", "科目名", "金額", "摘要"]
FIVE_KEYS = ["伝票日付", "借貸区分", "科目名", "金額", "摘要"]

def _list5_to_dict5(item):
    """5要素配列を5キー辞書にマッピング"""
    try:
        if not isinstance(item, list) or len(item) != 5:
            return None
        return dict(zip(FIVE_KEYS, [str(item[0]), str(item[1]), str(item[2]), str(item[3]), str(item[4])]))
    except Exception as e:
        logger.warning(f"配列→辞書変換失敗: {e}")
        return None

def _validate_and_normalize(entries: List[Dict]) -> List[Dict]:
    """エントリのバリデーションと正規化"""
    validated_data = []

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            logger.warning(f"エントリ{i}が辞書型ではありません: {type(entry)}")
            continue

        # 必須カラムの存在チェック
        missing_cols = [col for col in REQUIRED_5_COLUMNS if col not in entry]
        if missing_cols:
            logger.warning(f"エントリ{i}に必須カラムが不足: {missing_cols}")
            # 不足カラムを補完
            for col in missing_cols:
                entry[col] = "" if col != "金額" else 0

        # 金額の型チェック
        try:
            amount = entry.get("金額", 0)
            if isinstance(amount, str):
                amount = amount.replace(",", "").replace(" ", "")
            entry["金額"] = int(float(amount)) if amount else 0
        except (ValueError, TypeError):
            logger.warning(f"エントリ{i}の金額が無効: {entry.get('金額')}")
            entry["金額"] = 0

        # 仕訳Noの検証（任意フィールド）
        if "仕訳No" in entry:
            voucher_no = entry.get("仕訳No")
            if voucher_no is not None:
                # 文字列に変換
                voucher_no_str = str(voucher_no)
                # 妥当性チェック（1～6桁の数字）
                if re.match(r'^\d{1,6}$', voucher_no_str):
                    entry["仕訳No"] = voucher_no_str
                    logger.debug(f"エントリ{i}の仕訳No: {voucher_no_str}")
                else:
                    logger.warning(f"エントリ{i}の仕訳Noが無効（削除）: {voucher_no}")
                    # 無効な仕訳Noは削除せずNoneにする
                    entry["仕訳No"] = None

        validated_data.append(entry)

    return validated_data

def _get_fallback_entry() -> List[Dict]:
    """フォールバック用の1エントリ（空出力禁止）"""
    return [{
        "伝票日付": "",
        "借貸区分": "借方",
        "科目名": "抽出不能",
        "金額": 1,
        "摘要": "抽出不能【OCR注意:目視確認推奨】"
    }]

class JSONExtractionTimeout(Exception):
    """JSON抽出がタイムアウトした場合の例外"""
    pass


def _extract_json_with_linear_scan(
    text: str,
    start_pos: int = 0,
    timeout_seconds: float = 2.0
) -> Tuple[Optional[str], int]:
    """
    線形スキャンでJSON配列を抽出（catastrophic backtracking回避）

    Args:
        text: 検索対象テキスト
        start_pos: 検索開始位置
        timeout_seconds: タイムアウト時間（秒）

    Returns:
        Tuple[Optional[str], int]: (抽出されたJSON文字列, 次の検索開始位置)
                                    見つからない場合は (None, -1)

    Raises:
        JSONExtractionTimeout: タイムアウト時
    """
    start_time = time.perf_counter()
    text_len = len(text)

    # 最初の '[' を探す
    open_bracket_pos = text.find('[', start_pos)
    if open_bracket_pos == -1:
        return None, -1

    # ネスト深度と文字列リテラル状態を管理しながらスキャン
    depth = 0
    in_string = False
    escape_next = False
    i = open_bracket_pos

    while i < text_len:
        # タイムアウトチェック（100文字ごとに1回）
        if i % 100 == 0:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout_seconds:
                raise JSONExtractionTimeout(
                    f"JSON抽出が{timeout_seconds}秒を超えました "
                    f"(position={i}/{text_len})"
                )

        char = text[i]

        # エスケープシーケンス処理
        if escape_next:
            escape_next = False
            i += 1
            continue

        if char == '\\':
            escape_next = True
            i += 1
            continue

        # 文字列リテラルの開始/終了
        if char == '"':
            in_string = not in_string
            i += 1
            continue

        # 文字列リテラル内では括弧を無視
        if in_string:
            i += 1
            continue

        # 括弧のネスト管理
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1

            # depth==0 になったらJSON配列完成
            if depth == 0:
                json_str = text[open_bracket_pos:i+1]
                next_pos = i + 1
                return json_str, next_pos

        i += 1

    # 文字列終端まで到達したが対応する ']' が見つからない
    logger.debug(f"JSON配列が未完成: 開始位置={open_bracket_pos}, 最終depth={depth}")
    return None, -1


def extract_json_array_str(text: str, max_attempts: int = 10, timeout_seconds: float = 2.0) -> str:
    """
    テキストから最初のJSON配列文字列を抽出（線形スキャン方式）

    正規表現を使わずに線形スキャンでJSON配列を抽出することで、
    catastrophic backtrackingを回避し、安定した処理時間を保証します。

    Args:
        text: LLMの生レスポンステキスト
        max_attempts: 最大試行回数（複数の'['から試行）
        timeout_seconds: タイムアウト時間（秒）

    Returns:
        str: 抽出されたJSON配列文字列

    Raises:
        ValueError: JSON配列が見つからない場合
        JSONExtractionTimeout: タイムアウト時
    """
    overall_start = time.perf_counter()
    text = text.strip()

    # P2診断ログ: 入力テキストの基本情報
    text_len = len(text)
    open_count = text.count('[')
    close_count = text.count(']')
    logger.info(
        f"JSON抽出開始: テキスト長={text_len}文字, "
        f"'['={open_count}個, ']'={close_count}個"
    )
    logger.debug(f"先頭200文字: {text[:200]}")
    logger.debug(f"末尾200文字: {text[-200:]}")

    # コードブロックマーカーを除去（```json、```JSON、~~~対応）
    marker_found = False
    for marker in ["```json", "```JSON", "~~~json", "~~~JSON"]:
        if marker in text:
            start = text.find(marker) + len(marker)
            end_marker = "```" if marker.startswith("```") else "~~~"
            end = text.find(end_marker, start)
            if end != -1:
                text = text[start:end].strip()
                marker_found = True
                logger.debug(f"コードブロックマーカー除去: {marker}")
                break

    if not marker_found and "```" in text:
        start = text.find("```") + 3
        end = text.rfind("```")
        if end != -1 and end > start:
            text = text[start:end].strip()
            logger.debug("汎用コードブロックマーカー除去")

    # 線形スキャンで複数の '[' から試行
    attempt = 0
    current_pos = 0

    while attempt < max_attempts:
        # 全体タイムアウトチェック
        elapsed = time.perf_counter() - overall_start
        if elapsed > timeout_seconds:
            raise JSONExtractionTimeout(
                f"JSON抽出全体が{timeout_seconds}秒を超えました "
                f"(試行回数={attempt}/{max_attempts})"
            )

        try:
            # 線形スキャンで抽出
            json_str, next_pos = _extract_json_with_linear_scan(
                text,
                start_pos=current_pos,
                timeout_seconds=timeout_seconds - elapsed
            )

            if json_str is None:
                # これ以上 '[' が見つからない
                logger.debug(f"試行{attempt+1}: これ以上の'['が見つかりません")
                break

            # JSONとしてパース可能かチェック
            try:
                json.loads(json_str)
                # パース成功
                logger.info(
                    f"JSON配列抽出成功: {len(json_str)}文字 (試行{attempt+1}回目)"
                )
                return json_str
            except json.JSONDecodeError as e:
                # パース失敗 - 次の '[' から再試行
                logger.debug(
                    f"試行{attempt+1}: JSONパース失敗 ({e}), "
                    f"次の位置から再試行"
                )
                current_pos = next_pos
                attempt += 1
                continue

        except JSONExtractionTimeout:
            raise
        except Exception as e:
            logger.warning(f"試行{attempt+1}で予期しないエラー: {e}")
            attempt += 1
            current_pos += 1  # 1文字進めて再試行
            continue

    # すべての試行が失敗 - 角括弧補完を試行
    logger.warning(f"正規JSON配列が見つからない（{attempt}回試行）、角括弧補完を試行")

    # 開始角括弧がある場合、終了角括弧を補完
    if '[' in text and text.count('[') > text.count(']'):
        last_brace = text.rfind('}')
        if last_brace != -1:
            potential_json = text[text.find('['):last_brace + 1] + ']'
            try:
                json.loads(potential_json)
                logger.info("角括弧補完成功: 終了角括弧を追加")
                return potential_json
            except json.JSONDecodeError:
                logger.debug("終了角括弧補完も失敗")

    # 終了角括弧があるが開始角括弧がない場合
    if ']' in text and text.count(']') > text.count('['):
        first_brace = text.find('{')
        if first_brace != -1:
            potential_json = '[' + text[first_brace:]
            try:
                json.loads(potential_json)
                logger.info("角括弧補完成功: 開始角括弧を追加")
                return potential_json
            except json.JSONDecodeError:
                logger.debug("開始角括弧補完も失敗")

    raise ValueError(
        f"JSON配列パターンが見つかりません "
        f"({attempt}回試行、角括弧補完も失敗）"
    )

def parse_5cols_json(text: str) -> List[Dict]:
    """
    防御的5カラムJSONパーサー - 配列→辞書マッピング＋ゼロ件フォールバック対応

    Args:
        text: LLMの生レスポンステキスト

    Returns:
        List[Dict]: パースされた5カラムJSONリスト（最低1件保証）
    """
    parse_start = time.perf_counter()

    try:
        logger.info("防御的パーサー開始: LLMレスポンス解析")

        # 1. JSON文字列を抽出（タイムアウト保護付き）
        json_str = extract_json_array_str(text)
        logger.debug(f"抽出JSON文字列 (先頭500文字): {json_str[:500]}")
        logger.info(f"JSON文字列抽出完了: {len(json_str)}文字")

        # 2. JSONパース
        parsed_data = json.loads(json_str)
        logger.info(f"JSONパース成功: トップレベルの型 = {type(parsed_data)}")
        
        # 3. データ正規化（辞書・オブジェクト形式・配列形式すべて受け入れ）
        items = []
        
        if isinstance(parsed_data, dict) and "entries" in parsed_data and isinstance(parsed_data["entries"], list):
            # {"entries": [...]} 形式
            items = parsed_data["entries"]
            logger.info("検出: entries キー付きオブジェクト形式")
        elif isinstance(parsed_data, list):
            # 配列直接形式
            items = parsed_data
            logger.info("検出: 配列直接形式")
        else:
            logger.warning(f"予期しない形式: {type(parsed_data)}")
            items = []
        
        # 4. エントリ正規化（辞書・5要素配列の両方を受け入れ）
        normalized = []
        for i, item in enumerate(items):
            if isinstance(item, dict):
                # 既に辞書形式
                normalized.append(item)
                logger.debug(f"エントリ{i}: 辞書形式 - OK")
            elif isinstance(item, list) and len(item) == 5:
                # 5要素配列 → 辞書へマッピング
                mapped = _list5_to_dict5(item)
                if mapped:
                    normalized.append(mapped)
                    logger.info(f"エントリ{i}: 5要素配列 → 辞書変換成功")
                else:
                    logger.warning(f"エントリ{i}: 5要素配列の変換失敗")
            else:
                logger.warning(f"エントリ{i}が辞書型でも5要素配列でもありません: {type(item)}, length={len(item) if hasattr(item, '__len__') else 'N/A'}")
        
        # 5. バリデーション・正規化
        validated = _validate_and_normalize(normalized)
        
        # 6. ゼロ件フォールバック（空出力禁止）
        if not validated:
            logger.warning("有効エントリ0件 - フォールバック適用")
            validated = _get_fallback_entry()

        parse_elapsed = time.perf_counter() - parse_start
        logger.info(
            f"防御的パーサー完了: {len(validated)}エントリ確定 "
            f"(処理時間={parse_elapsed:.2f}秒)"
        )
        return validated

    except JSONExtractionTimeout as e:
        parse_elapsed = time.perf_counter() - parse_start
        logger.error(f"JSON抽出タイムアウト: {e} (経過時間={parse_elapsed:.2f}秒)")
        logger.debug(f"タイムアウト時テキスト（先頭500文字）: {text[:500]}")
        logger.debug(f"タイムアウト時テキスト（末尾500文字）: {text[-500:]}")
        return _get_fallback_entry()
    except json.JSONDecodeError as e:
        parse_elapsed = time.perf_counter() - parse_start
        logger.error(
            f"JSON抽出失敗 - パース失敗: {e} (経過時間={parse_elapsed:.2f}秒)"
        )
        logger.debug(f"失敗テキスト（先頭500文字）: {text[:500]}")
        return _get_fallback_entry()
    except Exception as e:
        parse_elapsed = time.perf_counter() - parse_start
        logger.error(
            f"防御的パーサーでエラー: {type(e).__name__}: {e} "
            f"(経過時間={parse_elapsed:.2f}秒)"
        )
        logger.debug(f"エラー時テキスト（先頭500文字）: {text[:500]}")
        return _get_fallback_entry()

def get_fallback_entry(error_msg: str = "抽出不能") -> List[Dict]:
    """
    フォールバック用の1要素配列を返す
    
    Args:
        error_msg: エラーメッセージ
        
    Returns:
        List[Dict]: フォールバック用の1要素配列
    """
    return [{
        "伝票日付": "",
        "借貸区分": "借方",
        "科目名": "抽出失敗",
        "金額": 1,
        "摘要": f"{error_msg}【OCR注意:目視確認推奨】"
    }]