"""골든셋 자동 채점 + 검색 방식 3종 비교.

  python eval/evaluate.py --compare              # dense/bm25/ensemble 비교 (LLM 비용 0)
  python eval/evaluate.py --sweep --mode dense   # 개발셋으로 임계값 튜닝, 홀드아웃으로 보고
  python eval/evaluate.py --retrieval-only       # 검색+거부만 채점
  python eval/evaluate.py                        # 전체 파이프라인 (LLM 키 필요)

발표에 쓸 숫자는 이 스크립트의 출력을 그대로 옮긴다.
모든 비율에 표본 수와 95% 신뢰구간을 붙인다. n 이 작을 때 비율만 말하는 것은
측정이 아니라 일화이기 때문이다.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from config import (ABSTAIN_THRESHOLD, DEFAULT_MODE, ENSEMBLE_ALPHA,  # noqa: E402
                    GOLDEN_SET, RETRIEVAL_MODES)
from retriever import search  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96):
    """이항 비율의 95% 신뢰구간. n 이 작을 때 정규근사보다 정직하다."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def pct(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k/n:.0%}, 95% CI {lo:.0%}-{hi:.0%})"


def pct_short(k: int, n: int) -> str:
    """표 안에 들어갈 압축형. 폭이 고정되어야 열이 맞는다."""
    if n == 0:
        return "n/a".center(20)
    lo, hi = wilson(k, n)
    return f"{k:>2}/{n:<2} {k/n:>4.0%} [{lo:>3.0%}-{hi:>4.0%}]"


def load_cases():
    return json.loads(Path(GOLDEN_SET).read_text(encoding="utf-8"))


def split(cases):
    """kind 별로 번갈아 나눠 개발셋/홀드아웃을 만든다.

    임계값을 고른 데이터로 그 임계값의 성적을 보고하면 과적합이 성능으로 둔갑한다.
    kind 별 층화를 하는 이유는 단순 홀짝 분할이면 함정 문항이 한쪽에 몰릴 수 있어서다.
    """
    dev, hold, seen = [], [], {}
    for c in cases:
        i = seen.get(c["kind"], 0)
        (dev if i % 2 == 0 else hold).append(c)
        seen[c["kind"]] = i + 1
    return dev, hold


def precompute(cases, mode, alpha):
    """검색을 문항당 한 번만 수행한다.

    검색 결과는 임계값과 무관하고 임계값은 거부 판정에만 쓰인다. 스윕이 임계값마다
    재검색하면 로컬 임베딩 모델을 수천 번 호출하게 되므로, 검색과 판정을 분리한다.
    """
    return [(c,) + search(c["question"], c["role"], mode, alpha) for c in cases]


def judge(pre, mode, alpha, threshold=None, retrieval_only=True):
    rows = []
    for case, hits, max_score, _blocked, backend in pre:
        thr = ABSTAIN_THRESHOLD[backend][mode] if threshold is None else threshold
        abstained = (not hits) or (max_score < thr)

        if retrieval_only:
            text = " ".join(h["text"] for h in hits)
        elif abstained:
            text = ""
        else:
            from answer import ask
            res = ask(case["question"], case["role"], mode=mode, alpha=alpha)
            abstained, text = res["abstained"], res.get("answer", "")

        top_doc = hits[0]["doc_id"] if hits else None
        if case["expect"] == "abstain":
            ok = abstained
        else:
            ok = (not abstained
                  and top_doc in case["gold_doc_ids"]
                  and all(k in text for k in case["must_include"]))
        rows.append({"id": case["id"], "kind": case["kind"], "expect": case["expect"],
                     "abstained": abstained, "top_doc": top_doc,
                     "max_score": round(max_score, 4), "ok": ok})
    return rows


def summarize(rows):
    ans = [r for r in rows if r["expect"] == "answer"]
    abs_ = [r for r in rows if r["expect"] == "abstain"]
    trap = [r for r in rows if r["kind"] == "trap"]
    return {
        "total": (sum(r["ok"] for r in rows), len(rows)),
        "answer": (sum(r["ok"] for r in ans), len(ans)),
        "abstain": (sum(r["ok"] for r in abs_), len(abs_)),
        "trap": (sum(r["ok"] for r in trap), len(trap)),
        "false_abstain": (sum(r["abstained"] for r in ans), len(ans)),
    }


def report(rows, title=""):
    if title:
        print(f"\n===== {title} =====")
    print(f"\n{'ID':<5}{'유형':<12}{'기대':<9}{'거부':<7}{'top문서':<14}{'점수':<9}{'결과'}")
    for r in rows:
        print(f"{r['id']:<5}{r['kind']:<12}{r['expect']:<9}"
              f"{str(r['abstained']):<7}{str(r['top_doc']):<14}"
              f"{r['max_score']:<9}{'PASS' if r['ok'] else 'FAIL'}")
    s = summarize(rows)
    print()
    print(f"정답률           : {pct(*s['total'])}")
    print(f"응답 정확도      : {pct(*s['answer'])}   (답이 있을 때 맞게 답한 비율)")
    print(f"거부 정확도      : {pct(*s['abstain'])}   (답이 없을 때 거부한 비율)")
    print(f"함정 거부율      : {pct(*s['trap'])}   (함정 문항만)")
    print(f"오거부(과잉거부) : {pct(*s['false_abstain'])}   (답이 있는데 거부한 비율)")
    return s


def tune(pre, mode, alpha, lo=0.02, hi=0.95, step=0.01):
    """개발셋에서만 임계값을 고른다. 동점이면 낮은 쪽(덜 거부하는 쪽)을 택한다."""
    best = []
    for thr in np.arange(lo, hi, step):
        rows = judge(pre, mode, alpha, float(thr))
        best.append((float(thr), sum(r["ok"] for r in rows) / len(rows)))
    best.sort(key=lambda x: (-x[1], x[0]))
    return best


def cmd_sweep(cases, mode, alpha):
    dev, hold = split(cases)
    best = tune(precompute(dev, mode, alpha), mode, alpha)
    thr, dev_acc = best[0]
    print(f"\n[{mode}] 임계값 스윕  개발셋 {len(dev)}문항 / 홀드아웃 {len(hold)}문항")
    print("  개발셋 상위 5개:")
    for t, a in best[:5]:
        print(f"    threshold={t:.2f}  정답률={a:.0%}")
    hold_rows = judge(precompute(hold, mode, alpha), mode, alpha, thr)
    s = summarize(hold_rows)
    hk, hn = s["total"]
    print(f"\n  선택한 임계값 : {thr:.2f}")
    print(f"  개발셋 정답률 : {dev_acc:.0%}  (이 값으로 임계값을 골랐으므로 과적합된 수치)")
    print(f"  홀드아웃 정답률: {pct(hk, hn)}  <- 발표에 쓸 숫자")
    if dev_acc - hk / hn > 0.15:
        print("  주의: 개발셋과 홀드아웃 차이가 15%p 를 넘는다. 임계값이 개발셋에 과적합됐다.")
    return thr


def cmd_compare(cases, alpha):
    """3방식 비교. 방식마다 개발셋으로 임계값을 따로 고른 뒤 홀드아웃으로 보고한다.

    방식별로 임계값을 따로 튜닝하지 않으면 한 방식에만 유리한 값으로 나머지를 재게 되어
    비교 자체가 성립하지 않는다.
    """
    dev, hold = split(cases)
    print(f"\n검색 방식 비교 (개발셋 {len(dev)}문항으로 임계값 튜닝 -> 홀드아웃 {len(hold)}문항으로 보고)")
    print(f"앙상블 alpha = {alpha}  (dense 가중치)\n")
    results = {}
    for mode in RETRIEVAL_MODES:
        thr = tune(precompute(dev, mode, alpha), mode, alpha)[0][0]
        rows = judge(precompute(hold, mode, alpha), mode, alpha, thr)
        results[mode] = (thr, summarize(rows), rows)

    head = f"{'방식':<10}{'임계값':>7}   {'정답률':^20} {'응답정확도':^20} {'함정거부':^20} {'오거부':^20}"
    print(head)
    print("-" * len(head))
    for mode, (thr, s, _) in results.items():
        print(f"{mode:<10}{thr:>7.2f}   {pct_short(*s['total'])} {pct_short(*s['answer'])} "
              f"{pct_short(*s['trap'])} {pct_short(*s['false_abstain'])}")
    print("\n(표기: 맞은수/전체  비율 [95% 신뢰구간].  오거부는 낮을수록 좋다)")

    best = max(results, key=lambda m: results[m][1]["total"][0])
    bk, bn = results[best][1]["total"]
    blo, bhi = wilson(bk, bn)
    print(f"\n최고 정답률: {best} ({bk}/{bn})")
    others = [m for m in results if m != best]
    overlap = [m for m in others
               if wilson(*results[m][1]["total"])[1] >= blo]
    if overlap:
        print(f"단 {', '.join(overlap)} 의 신뢰구간과 겹친다. n={bn} 에서는 "
              f"'{best} 가 더 낫다'고 단정할 수 없다.")
    else:
        print(f"다른 방식들의 신뢰구간과 겹치지 않는다. 차이가 표본 오차로 설명되지 않는다.")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--mode", default=DEFAULT_MODE, choices=RETRIEVAL_MODES)
    p.add_argument("--alpha", type=float, default=ENSEMBLE_ALPHA)
    a = p.parse_args()
    cases = load_cases()
    if a.compare:
        cmd_compare(cases, a.alpha)
    elif a.sweep:
        cmd_sweep(cases, a.mode, a.alpha)
    else:
        report(judge(precompute(cases, a.mode, a.alpha), a.mode, a.alpha,
                     retrieval_only=a.retrieval_only),
               f"{a.mode} (alpha={a.alpha})" if a.mode == "ensemble" else a.mode)
