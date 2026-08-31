"""권한 필터 전수 검사.

골든셋의 permission 문항 몇 개는 표본이라 신뢰구간이 붙는다. 하지만 권한 필터는
확률적 요소가 없는 결정적 로직이므로 전수 검사가 가능하다. 전수 검사에는 신뢰구간이
없다. 반례 0건이거나, 반례가 있거나 둘 중 하나다.

구현을 다시 쓰지 않고 retriever.search 를 그대로 호출하는 것이 중요하다. 검사용으로
권한 로직을 재구현하면 구현이 아니라 사본을 검사하게 되고, 진짜 구현의 버그는 그대로
남는다.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from config import DEFAULT_MODE, ROLE_LABEL, USERS, visible  # noqa: E402
from retriever import search  # noqa: E402
from store import load  # noqa: E402


def _live(meta):
    return meta.get("status", "active") == "active"


def main(mode=DEFAULT_MODE):
    ids, docs, metas = load()
    dist = Counter(m["clearance"] for m in metas if _live(m))
    live_n = sum(1 for m in metas if _live(m))
    print(f"청크 {len(ids)}개 중 live {live_n}개  등급 분포: " +
          ", ".join(f"{k} {v}" for k, v in sorted(dist.items())))
    print(f"검색 방식: {mode}\n")

    violations = []
    print(f"{'역할':<14}{'허용 청크':>10}{'차단 청크':>10}{'기대 차단':>10}   결과")
    for role in USERS:
        expect_allowed = sum(1 for m in metas if _live(m) and visible(m, role))
        expect_blocked = live_n - expect_allowed

        # 각 청크의 본문을 그대로 질의로 넣는다. 자기 자신이 1순위로 나오는 것이
        # 정상이므로, 권한이 없는 청크가 결과에 섞여 나오면 그것이 곧 유출이다.
        seen_blocked = None
        for i, text in enumerate(docs):
            hits, _, blocked, _ = search(text, role, mode, top_k=len(ids))
            seen_blocked = blocked
            for h in hits:
                if not _live(h) or not visible(h, role):
                    violations.append((role, ids[i], h["chunk_id"], h.get("clearance")))

        ok = (seen_blocked == expect_blocked)
        print(f"{ROLE_LABEL[role]:<14}{live_n-seen_blocked:>10}{seen_blocked:>10}"
              f"{expect_blocked:>10}   {'OK' if ok else 'MISMATCH'}")
        if not ok:
            violations.append((role, "-", "차단 청크 수 불일치", str(seen_blocked)))

    total = live_n * len(USERS)
    print(f"\n전수 검사 {total}조합 (live {live_n} x 페르소나 {len(USERS)})")
    if violations:
        print(f"반례 {len(violations)}건:")
        for v in violations[:20]:
            print("  ", v)
        return 1
    print("반례 0건. 권한 없는 청크가 검색 결과에 오른 경우가 한 번도 없음.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODE))
