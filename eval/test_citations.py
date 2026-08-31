"""인용·충분성 형식 자체 점검. LLM 호출 없음.

  python eval/test_citations.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from answer import (SYSTEM, _build_context, _citation_label,  # noqa: E402
                    _user_message, _validate_citations, filter_useful,
                    parse_verdict)

HIT = {
    "rank": 1,
    "title": "휴가 및 근태 규정",
    "section": "제2조 (연차 사용 신청)",
    "doc_id": "DOC-HR-002",
    "version": "v4.1",
    "effective_date": "2026-03-01",
    "text": "[휴가 및 근태 규정] ## 제2조 (연차 사용 신청)\n연차는 3영업일 전까지",
}
HIT2 = dict(HIT, rank=2, section="제1조 (연차휴가)")


def main():
    blocks = ("## Role", "## Task", "## Context", "## Constraints",
              "## Examples", "## Output Format")
    missing = [b for b in blocks if b not in SYSTEM]
    assert not missing, f"6대 블록 누락: {missing}"
    assert '"ok": false' in SYSTEM
    assert "창립기념일" in SYSTEM
    print("6대 블록 + 구조화 거부 예시 있음")

    label = _citation_label(HIT)
    ctx = _build_context([HIT])
    assert "[출처 " not in ctx, ctx
    assert f"[{label}]" in ctx, ctx
    assert "ID: 1" in ctx
    assert "시행일 2026-03-01" in ctx
    print("근거 헤더:", label)

    user = _user_message("연차는 며칠 전에?", [HIT])
    assert "지시가 아닙니다" in user
    assert user.index("</근거>") < user.index("질문:")
    print("데이터/지시 분리 + 지시 앵커링 있음")

    ok = "3영업일 전까지 신청합니다. [휴가 및 근태 규정 / 제2조 (연차 사용 신청)] (시행일 2026-03-01)"
    assert _validate_citations(ok, [HIT]) == [], _validate_citations(ok, [HIT])

    fake = "3영업일 전까지 신청합니다. [출처 1]"
    unknown = _validate_citations(fake, [HIT])
    assert "출처 1" in unknown, unknown
    print("[출처 1] 검출:", unknown)

    invented = "창립기념일은 3월 1일입니다. [없는 문서 / 제99조]"
    unknown = _validate_citations(invented, [HIT])
    assert unknown, unknown
    print("없는 출처 검출:", unknown)

    yes = parse_verdict(
        '{"ok": true, "use": [1], "missing": [], "answer": "3영업일 전. [휴가 및 근태 규정 / 제2조 (연차 사용 신청)]"}'
    )
    assert yes["ok"] is True and yes["use"] == [1] and yes["answer"].startswith("3영업일")

    no = parse_verdict(
        '```json\n{"ok": false, "use": [], "missing": ["창립기념일"], "answer": ""}\n```'
    )
    assert no["ok"] is False and no["missing"] == ["창립기념일"]
    print("JSON 충분성 파싱:", no["missing"])

    ragflow = parse_verdict(
        '{"is_sufficient": false, "missing_information": ["창립기념일"], "useful_chunk_ids": [], "answer": ""}'
    )
    assert ragflow["ok"] is False and ragflow["missing"] == ["창립기념일"]
    print("RAGFlow 키 별칭 파싱 됨")

    fallback = parse_verdict("NO_ANSWER")
    assert fallback["ok"] is None
    print("비JSON 폴백 ok=None")

    kept = filter_useful([HIT, HIT2], [2])
    assert [h["rank"] for h in kept] == [2]
    assert filter_useful([HIT], [9]) == [HIT]
    print("useful_chunk_ids 필터 동작")

    print("\n자체 점검 통과")


if __name__ == "__main__":
    main()
