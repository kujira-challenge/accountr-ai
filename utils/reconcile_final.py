# utils/reconcile_final.py
import re, unicodedata
from collections import defaultdict

def _nfkc(s): 
    return unicodedata.normalize("NFKC", s or "").strip()

def _amt(v):
    try:
        return int(str(v).replace(",", "").strip())
    except Exception:
        return 0

def _base_memo(m):
    t = _nfkc(m)
    # 摘要末尾の「; 借方: / ; 貸方:」の併記部分を落として共通部に
    t = t.split("; 借方:", 1)[0].split("; 貸方:", 1)[0]
    return re.sub(r"\s+", " ", t).strip()

def _side(row):
    # MJS45：どちらのコード欄が埋まっているかで判定（最優先）
    if row.get("（借）科目ｺｰﾄﾞ", ""): 
        return "D"
    if row.get("（貸）科目ｺｰﾄﾞ", ""): 
        return "C"
    # どちらも空なら（借貸区分）があればそれに従う
    val = row.get("借貸区分", "")
    return "D" if val == "借方" else ("C" if val == "貸方" else "N")

_BANK_WORDS = ("普通預金", "当座")
_CREDIT_SAFE = ("預り金", "敷金")  # 退去で頻出の貸方候補

def _has_bank(name: str) -> bool:
    n = name or ""
    return any(w in n for w in _BANK_WORDS)

def _is_credit_safe(name: str) -> bool:
    n = name or ""
    return any(w in n for w in _CREDIT_SAFE)

def reconcile_final(entries: list, *, drop_neutral: bool = True, sum_tolerance: int = 0) -> list:
    """
    最終整形（CSV直前）
      1) N行（両コード空）を除外（フォールバックは別UI通知）
      2) 左右スワップ：借方に銀行口座＆貸方が預り金/敷金 など安全パターンを是正
      3) one-vs-many（片側合算=反対側明細合計）→ 単独側を分割して揃える
      4) 同一サイド内で「max(金額) = sum(その他) ± tol」→ 合算行を1本落とす
    戻り値：整形済み entries
    """
    # ---- 1) N行除外 ----
    work = []
    for e in entries:
        if drop_neutral and not e.get("（借）科目ｺｰﾄﾞ","") and not e.get("（貸）科目ｺｰﾄﾞ",""):
            # CSVへは出さない（UIで別途警告）
            continue
        work.append(e)

    # グループ化（同一枠）= 日付 + 共通摘要
    buckets = defaultdict(list)
    for e in work:
        gid = (_nfkc(e.get("伝票日付","")), _base_memo(e.get("摘要","")))
        buckets[gid].append(e)

    out = []
    for (dte, memo_base), rows in buckets.items():
        # ---- 2) 左右スワップ（安全条件のみ）----
        D = [r for r in rows if _side(r) == "D"]
        C = [r for r in rows if _side(r) == "C"]

        # 借方に銀行口座（普通預金/当座）がいて、貸方側が預り金/敷金 等なら左右入替を試行
        if any(_has_bank(r.get("科目名","")) for r in D) and any(_is_credit_safe(r.get("科目名","")) for r in C):
            for r in rows:
                s = _side(r)
                if s == "D":
                    r["（貸）科目ｺｰﾄﾞ"], r["（借）科目ｺｰﾄﾞ"] = r.get("（借）科目ｺｰﾄﾞ",""), ""
                elif s == "C":
                    r["（借）科目ｺｰﾄﾞ"], r["（貸）科目ｺｰﾄﾞ"] = r.get("（貸）科目ｺｰﾄﾞ",""), ""
                r["摘要"] = _nfkc(r.get("摘要","")) + "【自動是正:左右入替】"
            # 再計算
            D = [r for r in rows if _side(r) == "D"]
            C = [r for r in rows if _side(r) == "C"]

        sumD = sum(_amt(r.get("金額",0)) for r in D)
        sumC = sum(_amt(r.get("金額",0)) for r in C)

        # ---- 3) one-vs-many 揃え込み（単独側を分割）----
        if len(D) == 1 and len(C) > 1 and abs(sumD - sumC) <= sum_tolerance:
            base = D[0]
            for c in C:
                nr = base.copy()
                nr["金額"] = str(_amt(c.get("金額",0)))
                nr["摘要"] = _nfkc(base.get("摘要","")) + "【分割(貸方に合わせ)】"
                out.append(nr)
            out.extend(C)
            continue

        if len(C) == 1 and len(D) > 1 and abs(sumD - sumC) <= sum_tolerance:
            base = C[0]
            for d in D:
                nr = base.copy()
                nr["金額"] = str(_amt(d.get("金額",0)))
                nr["摘要"] = _nfkc(base.get("摘要","")) + "【分割(借方に合わせ)】"
                out.append(nr)
            out.extend(D)
            continue

        # ---- 4) 同一サイド内の「max = sum(others)」合算落とし ----
        def drop_sum_row(rows_side: list) -> list:
            if len(rows_side) < 3:
                return rows_side
            amts = [_amt(r.get("金額",0)) for r in rows_side]
            mx = max(amts)
            if abs(mx - (sum(amts) - mx)) <= sum_tolerance:
                # 落とす候補（max 金額の行）から、科目名空 or 摘要に「合計/差額補完」を優先して1本除去
                cand = [r for r in rows_side if _amt(r.get("金額",0)) == mx]
                def score(row):
                    s = 0
                    if not row.get("科目名",""): s += 2
                    if ("合計" in row.get("摘要","")) or ("差額補完" in row.get("摘要","")): s += 1
                    return s
                drop = sorted(cand, key=score, reverse=True)[0]
                return [r for r in rows_side if r is not drop]
            return rows_side

        out.extend(drop_sum_row(D))
        out.extend(drop_sum_row(C))

    return out