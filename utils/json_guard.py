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
    テキストから最初のJSON配列文字列を抽出
    
    Args:
        text: LLMの生レスポンステキスト
        
    Returns:
        str: 抽出されたJSON配列文字列
        
    Raises:
        ValueError: JSON配列が見つからない場合
    """
    text = text.strip()
    
    # コードブロックマーカーを除去
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.rfind("```")
        if end != -1 and end > start:
            text = text[start:end].strip()
    
    # 最外層の [ ] を正規表現で抽出
    pattern = r'\[(?:[^[\]]*(?:\[[^\]]*\])*)*[^[\]]*\]'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        raise ValueError("JSON配列パターンが見つかりません")
    
    # 最初のマッチを返す
    json_str = matches[0].strip()
    logger.debug(f"JSON配列文字列を抽出: {len(json_str)}文字")
    
    return json_str

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
        
        # 3. リスト型チェック
        if not isinstance(parsed_data, list):
            raise ValueError(f"JSON配列である必要があります: {type(parsed_data)}")
        
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
        return validated_data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"JSONパースエラー: {e}")
    except Exception as e:
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