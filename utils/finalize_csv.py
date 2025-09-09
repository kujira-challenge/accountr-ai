# utils/finalize_csv.py
def finalize_csv_rows(csv_rows:list)->list:
    """
    MJS45行（コード列あり）に対する終段フィルタ
      - 両コード空はCSVから除外（フォールバックはUI表示のみ）
      - 完全重複の圧縮
    """
    clean, seen = [], set()
    for r in csv_rows:
        debit  = (r.get("（借）科目ｺｰﾄﾞ","") or "").strip()
        credit = (r.get("（貸）科目ｺｰﾄﾞ","") or "").strip()
        if debit=="" and credit=="":  # ← ★ここでだけ除外
            continue
        key = (r.get("伝票日付",""), r.get("金額",""), r.get("摘要",""), debit, credit)
        if key in seen: 
            continue
        seen.add(key)
        clean.append(r)
    return clean