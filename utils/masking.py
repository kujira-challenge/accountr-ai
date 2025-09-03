#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実名マスキング機能
ログ・UI表示用に個人識別情報をマスクする
CSVデータはマスクしない（業務データとして必要なため）
"""

import re
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

def mask_names(text: str) -> str:
    """
    日本語氏名らしきトークンをマスクする
    
    Args:
        text: 元のテキスト
        
    Returns:
        str: マスキング後のテキスト
    """
    if not text:
        return text
    
    # 2～6文字の日本語氏名パターンをマスク
    # 例: "山下良三" -> "山**", "田中" -> "田*"
    name_pattern = r'([一-龥ぁ-んァ-ヶ]{1,2})([一-龥ぁ-んァ-ヶ]{1,4})'
    text = re.sub(name_pattern, r'\1**', text)
    
    # カタカナ氏名のマスク
    # 例: "タナカハナコ" -> "タ****"
    katakana_name_pattern = r'([ア-ヶ]{1,2})([ア-ヶ]{2,6})'
    text = re.sub(katakana_name_pattern, r'\1**', text)
    
    # ひらがな氏名のマスク  
    # 例: "たなかはなこ" -> "た****"
    hiragana_name_pattern = r'([あ-ん]{1,2})([あ-ん]{2,6})'
    text = re.sub(hiragana_name_pattern, r'\1**', text)
    
    return text

def mask_personal_info(text: str) -> str:
    """
    個人識別情報を包括的にマスクする
    
    Args:
        text: 元のテキスト
        
    Returns:
        str: マスキング後のテキスト
    """
    if not text:
        return text
    
    # 氏名マスク
    text = mask_names(text)
    
    # 住所の番地マスク（番地詳細のみ、地名は保持）
    text = re.sub(r'(\d+)[-－](\d+)[-－](\d+)', r'***-***-***', text)
    
    # 電話番号マスク（中間部分のみ）
    text = re.sub(r'(\d{2,4})[-－](\d{2,4})[-－](\d{4})', r'\1-****-\3', text)
    
    return text

def mask_dict_values(data: Dict[str, Any], mask_keys: List[str] = None) -> Dict[str, Any]:
    """
    辞書の値をマスクする（指定されたキーのみ）
    
    Args:
        data: 元の辞書データ
        mask_keys: マスクするキーのリスト（Noneの場合はデフォルトキー使用）
        
    Returns:
        Dict: マスクされた辞書データ
    """
    if mask_keys is None:
        mask_keys = ["摘要", "備考", "契約者名", "オーナー"]
    
    masked_data = data.copy()
    
    for key in mask_keys:
        if key in masked_data and isinstance(masked_data[key], str):
            masked_data[key] = mask_personal_info(masked_data[key])
    
    return masked_data

def mask_list_for_logging(data_list: List[Dict], sample_size: int = 2) -> List[Dict]:
    """
    ログ出力用にリストデータをマスクする（サンプル件数制限付き）
    
    Args:
        data_list: 元のデータリスト
        sample_size: ログに出力するサンプル件数
        
    Returns:
        List[Dict]: マスクされたサンプルデータ
    """
    if not data_list:
        return []
    
    # サンプル件数制限
    sample_data = data_list[:sample_size]
    
    # 各エントリをマスク
    masked_samples = []
    for entry in sample_data:
        if isinstance(entry, dict):
            masked_entry = mask_dict_values(entry)
            masked_samples.append(masked_entry)
        else:
            masked_samples.append(entry)
    
    return masked_samples