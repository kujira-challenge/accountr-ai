# utils/reconcile_entries.py
import re, unicodedata
from collections import defaultdict

def _nfkc(s): return unicodedata.normalize("NFKC", s or "").strip()
def _amt(v):
    try: return int(str(v).replace(",","").strip())
    except: return 0
def _base_memo(m):
    t = _nfkc(m)
    t = t.split("; 借方:",1)[0].split("; 貸方:",1)[0]
    return re.sub(r"\s+"," ", t).strip()

_BANK = ("普通預金","当座")
_CREDIT_SAFE = ("預り金","敷金")

def _side_of(entry):  # 5カラムは借貸区分のみで判定
    s = entry.get("借貸区分","")
    return "D" if s=="借方" else ("C" if s=="貸方" else "N")

def reconcile_entries(entries:list, *, sum_tolerance:int=0)->list:
    """
    5カラムJSON向けの"形の整形"のみを行う。
      - one-vs-many(片側合算=反対側明細合計) → 合算側を分割
      - 同一サイド内「max = sum(others)±tol」→ 合算行を1本落とす
      - 安全条件の左右入替（借方=普通預金/当座 & 貸方=預り金/敷金 など）
    科目コード列は一切参照しない（ここで両コード空の除去はしない）。
    """
    buckets = defaultdict(list)
    for e in entries:
        gid = (_nfkc(e.get("伝票日付","")), _base_memo(e.get("摘要","")))
        buckets[gid].append(e)

    out = []
    for _, rows in buckets.items():
        D = [r for r in rows if _side_of(r)=="D"]
        C = [r for r in rows if _side_of(r)=="C"]

        # 2-1) 安全な左右入替（"名前"ベース）
        if any(any(w in (r.get("科目名","") or "") for w in _BANK) for r in D) and \
           any(any(w in (r.get("科目名","") or "") for w in _CREDIT_SAFE) for r in C):
            for r in rows:
                s = _side_of(r)
                if s=="D": r["借貸区分"]="貸方"
                elif s=="C": r["借貸区分"]="借方"
                r["摘要"] = _nfkc(r.get("摘要","")) + "【自動是正:左右入替】"
            D = [r for r in rows if _side_of(r)=="D"]
            C = [r for r in rows if _side_of(r)=="C"]

        sumD = sum(_amt(r.get("金額",0)) for r in D)
        sumC = sum(_amt(r.get("金額",0)) for r in C)

        # 2-2) one-vs-many 揃え込み（合算→明細に分割）
        if len(D)==1 and len(C)>1 and abs(sumD-sumC)<=sum_tolerance:
            base = D[0]
            for c in C:
                nr = base.copy()
                nr["金額"] = str(_amt(c.get("金額",0)))
                nr["摘要"] = _nfkc(base.get("摘要","")) + "【分割(貸方に合わせ)】"
                out.append(nr)
            out.extend(C);  continue

        if len(C)==1 and len(D)>1 and abs(sumD-sumC)<=sum_tolerance:
            base = C[0]
            for d in D:
                nr = base.copy()
                nr["金額"] = str(_amt(d.get("金額",0)))
                nr["摘要"] = _nfkc(base.get("摘要","")) + "【分割(借方に合わせ)】"
                out.append(nr)
            out.extend(D);  continue

        # 2-3) 同一サイド内「max = sum(others)」合算落とし
        def drop_sum(rows_side):
            if len(rows_side) < 3: return rows_side
            amts = [_amt(r["金額"]) for r in rows_side]
            mx = max(amts)
            if abs(mx - (sum(amts)-mx)) <= sum_tolerance:
                cand = [r for r in rows_side if _amt(r["金額"])==mx]
                def score(r):
                    s=0
                    if not r.get("科目名",""): s+=2
                    if "合計" in r.get("摘要","") or "差額補完" in r.get("摘要",""): s+=1
                    return s
                drop = sorted(cand, key=score, reverse=True)[0]
                return [r for r in rows_side if r is not drop]
            return rows_side

        out.extend(drop_sum(D)); out.extend(drop_sum(C))
    return out