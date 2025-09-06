# utils/reconcile_dedupe.py
import re
import unicodedata
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def _nfkc(s):
    return unicodedata.normalize("NFKC", s or "").strip()

def _amt(v):
    try:
        return int(str(v).replace(",", "").strip())
    except:
        return 0

def _base_memo(m):
    """摘要の共通部分を抽出（「; 借方:」「; 貸方:」より前）+ より広い正規化"""
    t = _nfkc(m)
    t = t.split("; 借方:", 1)[0].split("; 貸方:", 1)[0]
    t = re.sub(r"\s+", " ", t).strip()
    
    # 共通のキーワードパターンを抽出（住所・物件名など）
    # 「合計」「詳細N」などのサフィックスを除去して共通部分を抽出
    t = re.sub(r'\s+(合計|詳細\d*|明細\d*)$', '', t)
    t = re.sub(r'\s+\d+$', '', t)  # 末尾の数字も除去
    
    return t.strip()

def _side(row):
    """MJS45: どちらのコード欄が埋まっているかで判定"""
    return "D" if row.get("（借）科目ｺｰﾄﾞ", "") else ("C" if row.get("（貸）科目ｺｰﾄﾞ", "") else "N")

def reconcile_and_dedupe(entries: list) -> list:
    """
    重複見えを解消する最終整形：
      1) N行（両コード空）除去（フォールバックを除く）
      2) one-vs-many（合算=明細合計）を分割して揃える
      3) 同一サイド内の 合算=max=sum(others) を除去
      4) そのほかは通過
    """
    if not entries:
        return entries
    
    original_count = len(entries)
    logger.info(f"Starting reconciliation and deduplication with {original_count} entries")
    
    # ---- 1) N行除去（フォールバックは別ハンドリング） ----
    work = []
    n_removed = 0
    for e in entries:
        if e.get("（借）科目ｺｰﾄﾞ", "") == "" and e.get("（貸）科目ｺｰﾄﾞ", "") == "":
            n_removed += 1
            logger.debug(f"Removing empty code row: {e.get('摘要', '')[:50]}...")
        else:
            work.append(e)
    
    if n_removed > 0:
        logger.info(f"Removed {n_removed} rows with empty both-side codes")
    
    # グループ化キー
    buckets = defaultdict(list)
    for e in work:
        gid = (_nfkc(e.get("伝票日付", "")), _base_memo(e.get("摘要", "")))
        buckets[gid].append(e)
    
    out = []
    one_vs_many_splits = 0
    sum_rows_removed = 0
    
    for gid, rows in buckets.items():
        # サイド分割
        D = [r for r in rows if _side(r) == "D"]
        C = [r for r in rows if _side(r) == "C"]
        
        sumD = sum(_amt(r.get("金額", 0)) for r in D)
        sumC = sum(_amt(r.get("金額", 0)) for r in C)
        
        logger.debug(f"Group {gid}: D={len(D)} (sum={sumD}), C={len(C)} (sum={sumC})")
        
        # ---- 2) 同一サイド内の合算行を削除（len>=3 & max==sum(others))を先に実行 ----
        def drop_sum_row(rows_side, side_name):
            if len(rows_side) < 3:
                return rows_side
            amts = [_amt(r["金額"]) for r in rows_side]
            mx = max(amts)
            if mx == sum(amts) - mx:
                # 候補から1行落とす（摘要に「合計/差額補完」or 科目名空を優先）
                cand = [r for r in rows_side if _amt(r["金額"]) == mx]
                
                def score(r):
                    s = 0
                    if r.get("科目名", "") == "":
                        s += 2
                    memo = r.get("摘要", "")
                    if "合計" in memo or "差額補完" in memo:
                        s += 1
                    return s
                
                drop = sorted(cand, key=score, reverse=True)[0]
                rows_side = [r for r in rows_side if r is not drop]
                logger.info(f"Removed sum row in {side_name}: amount={mx}")
                nonlocal sum_rows_removed
                sum_rows_removed += 1
            return rows_side
        
        # 合算行除去を最初に実行
        D2 = drop_sum_row(D, "debit")
        C2 = drop_sum_row(C, "credit")
        
        # 合算行除去後の合計を再計算
        sumD2 = sum(_amt(r.get("金額", 0)) for r in D2)
        sumC2 = sum(_amt(r.get("金額", 0)) for r in C2)
        
        # ---- 3) one-vs-many 分割（合算行除去後のデータで実行） ----
        if len(D2) == 1 and len(C2) > 1 and sumD2 == sumC2:
            # 借方1本を貸方の本数に分割
            d = D2[0]
            amounts = [_amt(c["金額"]) for c in C2]
            logger.info(f"One-vs-many split: splitting debit {sumD2} into {len(amounts)} parts")
            one_vs_many_splits += 1
            
            for a in amounts:
                nr = d.copy()
                nr["金額"] = str(a)
                nr["摘要"] = _nfkc(d.get("摘要", "")) + " 【分割(貸方に合わせ)】"
                out.append(nr)
            out.extend(C2)  # 貸方はそのまま
            continue
            
        if len(C2) == 1 and len(D2) > 1 and sumC2 == sumD2:
            # 貸方1本を借方の本数に分割
            c = C2[0]
            amounts = [_amt(d["金額"]) for d in D2]
            logger.info(f"One-vs-many split: splitting credit {sumC2} into {len(amounts)} parts")
            one_vs_many_splits += 1
            
            for a in amounts:
                nr = c.copy()
                nr["金額"] = str(a)
                nr["摘要"] = _nfkc(c.get("摘要", "")) + " 【分割(借方に合わせ)】"
                out.append(nr)
            out.extend(D2)
            continue
        
        # ---- 4) その他の場合は合算行除去後のデータをそのまま追加 ----
        out.extend(D2)
        out.extend(C2)
    
    final_count = len(out)
    total_removed = original_count - final_count
    
    if one_vs_many_splits > 0:
        logger.info(f"Performed {one_vs_many_splits} one-vs-many splits")
    if sum_rows_removed > 0:
        logger.info(f"Removed {sum_rows_removed} sum rows")
    if total_removed > 0:
        logger.info(f"Total entries changed: {original_count} -> {final_count} (net change: {total_removed})")
    
    return out