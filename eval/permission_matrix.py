"""권한 필터 전수 검사.

골든셋의 permission 문항 몇 개는 표본이라 신뢰구간이 붙는다. 하지만 권한 필터는
확률적 요소가 없는 결정적 로직이므로 전수 검사가 가능하다. 전수 검사에는 신뢰구간이
없다. 반례 0건이거나, 반례가 있거나 둘 중 하나다.

청크 32개 x 역할 3개 = 96조합을 모두 확인한다.

구현을 다시 쓰지 않고 retriever.search 를 그대로 호출하는 것이 중요하다. 검사용으로
권한 로직을 재구현하면 구현이 아니라 사본을 검사하게 되고, 진짜 구현의 버그는 그대로
남는다.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from config import DEFAULT_MODE, ROLE_CLEARANCE, ROLE_LABEL  # noqa: E402
from retriever import search  # noqa: E402
from store import load  # noqa: E402


def main(mode=DEFAULT_MODE):
    ids, docs, metas = load()
    dist = Counter(m["clearance"] for m in metas)
    print(f"청크 {len(ids)}개  등급 분포: " +
          ", ".join(f"{k} {v}" for k, v in sorted(dist.items())))
    print(f"검색 방식: {mode}\n")

    violations = []
    print(f"{'역할':<12}{'허용 청크':>10}{'차단 청크':>10}{'기대 차단':>10}   결과")
    for role, allowed in ROLE_CLEARANCE.items():
        expect_allowed = sum(v for k, v in dist.items() if k in allowed)
        expect_blocked = len(ids) - expect_allowed

        # 각 청크의 본문을 그대로 질의로 넣는다. 자기 자신이 1순위로 나오는 것이
        # 정상이므로, 권한이 없는 청크가 결과에 섞여 나오면 그것이 곧 유출이다.
        seen_blocked = None
        for i, text in enumerate(docs):
            hits, _, blocked, _ = search(text, role, mode, top_k=len(ids))
            seen_blocked = blocked
            for h in hits:
                if h["clearance"] not in allowed:
                    violations.append((role, ids[i], h["chunk_id"], h["clearance"]))

        ok = (seen_blocked == expect_blocked)
        print(f"{ROLE_LABEL[role]:<12}{len(ids)-seen_blocked:>10}{seen_blocked:>10}"
              f"{expect_blocked:>10}   {'OK' if ok else 'MISMATCH'}")
        if not ok:
            violations.append((role, "-", "차단 청크 수 불일치", str(seen_blocked)))

    total = len(ids) * len(ROLE_CLEARANCE)
    print(f"\n전수 검사 {total}조합 (청크 {len(ids)} x 역할 {len(ROLE_CLEARANCE)})")
    if violations:
        print(f"반례 {len(violations)}건:")
        for v in violations[:20]:
            print("  ", v)
        return 1
    print("반례 0건. 권한 없는 청크가 검색 결과에 오른 경우가 한 번도 없음.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODE))
