# utils/finalize_csv.py
import logging

logger = logging.getLogger(__name__)

def finalize_csv_rows(csv_rows: list) -> list:
    """
    MJS45行（コード列あり）に対する終段フィルタ（CSV直前の最終処理）
      - 両コード空はCSVから除外（フォールバックはUI表示のみ）
      - 完全重複の圧縮
    
    Args:
        csv_rows: MJS45形式の辞書リスト
    
    Returns:
        list: フィルタ後のMJS45行リスト
    """
    if not csv_rows:
        logger.warning("finalize_csv_rows: No input rows to process")
        return []
    
    original_count = len(csv_rows)
    logger.info(f"Starting final CSV processing: {original_count} rows")
    
    clean, seen = [], set()
    empty_code_count = 0
    duplicate_count = 0
    
    for r in csv_rows:
        debit  = (r.get("（借）科目ｺｰﾄﾞ","") or "").strip()
        credit = (r.get("（貸）科目ｺｰﾄﾞ","") or "").strip()
        
        # 両コード空チェック（ここでのみ除外）
        if debit=="" and credit=="":
            empty_code_count += 1
            logger.debug(f"Excluding empty codes row: {r.get('摘要', '')[:50]}...")
            continue
        
        # 重複チェック
        key = (r.get("伝票日付",""), r.get("金額",""), r.get("摘要",""), debit, credit)
        if key in seen:
            duplicate_count += 1
            logger.debug(f"Excluding duplicate row: {r.get('摘要', '')[:50]}...")
            continue
        
        seen.add(key)
        clean.append(r)
    
    final_count = len(clean)
    
    # 詳細ログ出力
    logger.info(f"Final CSV processing completed:")
    logger.info(f"  Input rows: {original_count}")
    logger.info(f"  Empty codes excluded: {empty_code_count}")
    logger.info(f"  Duplicates excluded: {duplicate_count}")
    logger.info(f"  Final output rows: {final_count}")
    
    if final_count == 0:
        logger.error("WARNING: Final CSV has 0 rows after processing!")
    
    return clean