# utils/postprocess_dedupe.py
import re
import unicodedata
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()

def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def common_memo_base(memo: str) -> str:
    """
    摘要の '; 借方:' / '; 貸方:' 以降を落として共通部のみ返す
    """
    t = (memo or "")
    t = t.split("; 借方:", 1)[0].split("; 貸方:", 1)[0]
    return _collapse_ws(_nfkc(t))

_MICRO_PATTERNS = [
    r"No\.\s*[\w\-\/]+",     # No.E-2, No.1/2, No.32 など
    r"\b\d{1,2}月分\b",      # 1月分, 12月分
    r"\b\d{1,3}号\b",        # 101号 等（あれば）
]

def micro_id(memo_base: str) -> str:
    """
    微識別子（No.○/○、○月分 など）を抽出し、空白レスで結合
    """
    s = memo_base or ""
    toks = []
    for pat in _MICRO_PATTERNS:
        toks += re.findall(pat, s, flags=re.IGNORECASE)
    j = "|".join(_nfkc(t).replace(" ", "") for t in toks)
    return j

def _amount(v) -> int:
    try:
        return int(str(v).replace(",", "").strip())
    except Exception:
        return 0

def deduplicate_entries(entries: list, sum_tolerance: int = 0) -> list:
    """
    5カラムJSON（dict）配列に対して重複を除去する。
    ルール:
      1) 完全重複（全キー一致）は1件に
      2) 同一グループ（= 日付 + 共通摘要 + 微識別子）内で、
         各サイドごとに「max(金額) == sum(他) ± tol」なら max(金額) 行を削除（= 合算行を落とす）
    返り値: クリーンな entries
    """
    if not entries:
        return entries
    
    original_count = len(entries)
    logger.debug(f"Starting deduplication with {original_count} entries")
    
    # 1) 正規化しつつグループ化
    groups = defaultdict(list)
    normed = []
    for e in entries:
        d = {
            "伝票日付": _nfkc(e.get("伝票日付", "")),
            "借貸区分": e.get("借貸区分", ""),
            "科目名":   _nfkc(e.get("科目名", "")),
            "金額":     str(_amount(e.get("金額", 0))),
            "摘要":     _nfkc(e.get("摘要", "")),
        }
        base = common_memo_base(d["摘要"])
        gid = (d["伝票日付"], base, micro_id(base))
        d["_gid"] = gid
        normed.append(d)
        groups[gid].append(d)

    # 2) 完全重複の除去
    unique_by_all_fields = []
    seen = set()
    def _key_all(x):
        return (x["伝票日付"], x["借貸区分"], x["科目名"], x["金額"], x["摘要"])
    
    duplicate_count = 0
    for d in normed:
        k = _key_all(d)
        if k in seen: 
            duplicate_count += 1
            continue
        seen.add(k)
        unique_by_all_fields.append(d)
    
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} complete duplicates (5-column match)")

    # 3) 合算行の除去（サイド別）
    out = []
    by_gid = defaultdict(list)
    for d in unique_by_all_fields:
        by_gid[d["_gid"]].append(d)

    sum_removed_debit = 0
    sum_removed_credit = 0
    
    for gid, rows in by_gid.items():
        # サイドごとに処理
        keep = rows[:]
        for side in ("借方", "貸方"):
            side_rows = [r for r in rows if r["借貸区分"] == side]
            if len(side_rows) >= 3:
                amts = [_amount(r["金額"]) for r in side_rows]
                mx = max(amts)
                others_sum = sum(amts) - mx
                if abs(mx - others_sum) <= int(sum_tolerance):
                    # 合算（最大）と思しき1行を落とす：候補が複数あるなら「摘要に '合計' 無し」「科目名が空」を優先して落とす
                    # （汎用的なヒューリスティック）
                    candidates = [r for r in side_rows if _amount(r["金額"]) == mx]
                    def score(r):
                        s = 0
                        if "合計" in r["摘要"]: s -= 1
                        if r["科目名"] == "": s += 1
                        return s
                    drop = sorted(candidates, key=score, reverse=True)[0]
                    keep = [r for r in keep if r is not drop]
                    if side == "借方":
                        sum_removed_debit += 1
                    else:
                        sum_removed_credit += 1
                    logger.debug(f"Removed sum line in {side} for group {gid}: amount={mx}")
        out.extend(keep)

    if sum_removed_debit > 0 or sum_removed_credit > 0:
        logger.info(f"Removed sum lines - debit: {sum_removed_debit}, credit: {sum_removed_credit}")

    # 4) 作業用キーを剥がす
    for r in out:
        r.pop("_gid", None)
    
    final_count = len(out)
    total_removed = original_count - final_count
    if total_removed > 0:
        logger.info(f"Total entries removed in deduplication: {total_removed} (from {original_count} to {final_count})")
    
    return out