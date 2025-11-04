#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON抽出ガード機能
LLM生応答から安全にJSON配列を抽出・パースする
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional

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

def extract_json_array_str(text: str) -> str:
    """
    テキストから最初のJSON配列文字列を抽出（角括弧補完機能付き）
    
    Args:
        text: LLMの生レスポンステキスト
        
    Returns:
        str: 抽出されたJSON配列文字列
        
    Raises:
        ValueError: JSON配列が見つからない場合
    """
    text = text.strip()
    
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
                break
    
    if not marker_found and "```" in text:
        start = text.find("```") + 3
        end = text.rfind("```")
        if end != -1 and end > start:
            text = text[start:end].strip()
    
    # 最外層の [ ] を正規表現で抽出
    pattern = r'\[(?:[^[\]]*(?:\[[^\]]*\])*)*[^[\]]*\]'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        json_str = matches[0].strip()
        logger.debug(f"JSON配列文字列を抽出: {len(json_str)}文字")
        return json_str
    
    # 角括弧補完を試行（1個だけ不足の場合）
    logger.debug("正規JSON配列が見つからないため、角括弧補完を試行")
    
    # 開始角括弧がある場合、終了角括弧を補完
    if '[' in text and text.count('[') > text.count(']'):
        # 最後の ']' を見つけて、その後に ']' を追加
        last_brace = text.rfind('}')
        if last_brace != -1:
            potential_json = text[:last_brace + 1] + ']'
            # パターンマッチング再実行
            matches = re.findall(pattern, potential_json, re.DOTALL)
            if matches:
                logger.info("角括弧補完成功: 終了角括弧を追加")
                return matches[0].strip()
    
    # 終了角括弧があるが開始角括弧がない場合
    if ']' in text and text.count(']') > text.count('['):
        # 最初の '{' の前に '[' を追加
        first_brace = text.find('{')
        if first_brace != -1:
            potential_json = '[' + text
            matches = re.findall(pattern, potential_json, re.DOTALL)
            if matches:
                logger.info("角括弧補完成功: 開始角括弧を追加")
                return matches[0].strip()
    
    raise ValueError("JSON配列パターンが見つかりません（角括弧補完も失敗）")

def parse_5cols_json(text: str) -> List[Dict]:
    """
    防御的5カラムJSONパーサー - 配列→辞書マッピング＋ゼロ件フォールバック対応
    
    Args:
        text: LLMの生レスポンステキスト
        
    Returns:
        List[Dict]: パースされた5カラムJSONリスト（最低1件保証）
    """
    try:
        logger.info("防御的パーサー開始: Claude APIレスポンス解析")
        
        # 1. JSON文字列を抽出
        json_str = extract_json_array_str(text)
        logger.debug(f"抽出JSON文字列 (先頭500文字): {json_str[:500]}")
        
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
        
        logger.info(f"防御的パーサー完了: {len(validated)}エントリ確定")
        return validated
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON抽出失敗 - パース失敗: {e}")
        logger.debug(f"失敗テキスト（先頭500文字）: {text[:500]}")
        return _get_fallback_entry()
    except Exception as e:
        logger.error(f"防御的パーサーでエラー: {type(e).__name__}: {e}")
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