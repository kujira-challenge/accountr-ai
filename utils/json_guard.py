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
    LLM生応答から5カラムJSON配列を安全に抽出・パース
    
    Args:
        text: LLMの生レスポンステキスト
        
    Returns:
        List[Dict]: パースされた5カラムJSONリスト
        
    Raises:
        ValueError: パースに失敗した場合
        KeyError: 必須カラムが不足している場合
    """
    try:
        # 1. JSON配列文字列を抽出
        json_str = extract_json_array_str(text)
        
        # 2. JSONパース
        parsed_data = json.loads(json_str)
        
        # DEBUG: 実際のClaude APIレスポンス構造をログ出力
        logger.error(f"DEBUG - parsed_data type: {type(parsed_data)}")
        if isinstance(parsed_data, list) and len(parsed_data) > 0:
            logger.error(f"DEBUG - first element type: {type(parsed_data[0])}")
            logger.error(f"DEBUG - first element content: {parsed_data[0]}")
            if isinstance(parsed_data[0], list) and len(parsed_data[0]) > 0:
                logger.error(f"DEBUG - first nested element: {parsed_data[0][0]}")
        
        # 3. リスト型チェック
        if not isinstance(parsed_data, list):
            raise ValueError(f"JSON配列である必要があります: {type(parsed_data)}")
        
        # HOTFIX: If Claude API returns nested array, flatten it
        if isinstance(parsed_data, list) and len(parsed_data) > 0 and isinstance(parsed_data[0], list):
            logger.warning("Detected nested array structure - flattening...")
            flattened_data = []
            for subarray in parsed_data:
                if isinstance(subarray, list):
                    flattened_data.extend(subarray)
                else:
                    flattened_data.append(subarray)
            parsed_data = flattened_data
            logger.info(f"Flattened array: {len(parsed_data)} entries")
        
        # 4. 各要素の5カラムチェック
        validated_data = []
        for i, entry in enumerate(parsed_data):
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
            
            validated_data.append(entry)
        
        logger.info(f"5カラムJSONパース成功: {len(validated_data)}エントリ")
        
        # デバッグ用：LLM生応答の冒頭末尾を保存（最大8KB）
        sanitized_text = text[:8000] if len(text) > 8000 else text
        logger.debug(f"LLM生応答（冒頭～末尾8KB）: {sanitized_text}")
        
        return validated_data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON抽出失敗 - 原因: JSONパース失敗 - {e}")
        logger.debug(f"パース失敗テキスト（先頭500文字）: {text[:500]}...")
        raise ValueError(f"JSONパースエラー: {e}")
    except Exception as e:
        logger.error(f"JSON抽出失敗 - 原因: その他エラー - {type(e).__name__}: {e}")
        logger.debug(f"エラー発生テキスト（先頭500文字）: {text[:500]}...")
        raise ValueError(f"5カラムJSON抽出エラー: {e}")

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
        "借貸区分": "",
        "科目名": "",
        "金額": 0,
        "摘要": f"{error_msg}【OCR注意:目視確認推奨】"
    }]