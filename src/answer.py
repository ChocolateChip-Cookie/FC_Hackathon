"""신뢰도 레이어. 세 겹으로 막는다.

1) 검색 점수가 임계값 미만이면 LLM을 호출하지도 않고 거부한다 (결정적, 비용 0)
2) 프롬프트로 근거 밖 추론을 금지하고, 근거가 부족하면 NO_ANSWER를 내도록 강제한다
3) 생성된 답변의 인용을 실제 검색 결과와 대조해 없는 출처를 지어낸 경우 걸러낸다
"""
import re

from config import ABSTAIN_THRESHOLD, DEFAULT_MODE, ENSEMBLE_ALPHA
from llm import complete
from retriever import log_access, search, should_abstain

ABSTAIN_MSG = ("사내 규정 문서에서 근거를 찾지 못했습니다. "
               "정확한 답변을 위해 담당 부서에 문의해 주세요.")

SYSTEM = """당신은 사내 규정 안내 어시스턴트입니다. 아래 규칙을 반드시 지키십시오.

1. 아래 <근거> 안의 내용만 사용해 답하십시오. 일반 상식이나 다른 회사의 관행을 끌어오지 마십시오.
2. 모든 사실 문장 끝에 출처를 [문서명 / 섹션] 형식으로 붙이십시오.
3. <근거>만으로 답할 수 없거나 근거가 질문과 무관하면, 다른 말 없이 정확히 NO_ANSWER 라고만 출력하십시오.
4. 추측, 가정, "일반적으로는" 같은 표현을 쓰지 마십시오.
5. 한국어로 간결하게 답하십시오."""


def _build_context(hits):
    blocks = []
    for h in hits:
        blocks.append(
            f"[출처 {h['rank']}] 문서: {h['title']} ({h['doc_id']}, {h['version']}, "
            f"시행일 {h['effective_date']})\n섹션: {h['section']}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def _validate_citations(text, hits):
    """답변이 인용한 섹션명이 실제 검색 결과에 있는지 확인한다."""
    cited = re.findall(r"\[([^\[\]]+?)\]", text)
    known = {h["section"] for h in hits} | {h["title"] for h in hits}
    unknown = []
    for c in cited:
        if not any(k in c or c in k for k in known):
            unknown.append(c)
    return unknown


def ask(query: str, role: str, mode: str = DEFAULT_MODE, alpha: float = ENSEMBLE_ALPHA):
    hits, max_score, blocked, backend = search(query, role, mode, alpha)

    result = {"query": query, "role": role, "hits": hits, "max_score": max_score,
              "blocked_chunks": blocked, "backend": backend, "mode": mode,
              "abstained": False, "reason": None, "unknown_citations": []}

    # 1겹: 검색 단계 거부
    if not hits or should_abstain(max_score, backend, mode):
        result.update(abstained=True, answer=ABSTAIN_MSG,
                      reason=f"검색 최고 유사도 {max_score:.3f} < 임계값 "
                             f"{ABSTAIN_THRESHOLD[backend][mode]:.2f}")
        log_access(role, query, hits, True, blocked, mode)
        return result

    # 2겹: 생성 단계 거부
    raw = complete(SYSTEM, f"<근거>\n{_build_context(hits)}\n</근거>\n\n질문: {query}").strip()
    if "NO_ANSWER" in raw:
        result.update(abstained=True, answer=ABSTAIN_MSG, reason="모델이 근거 부족으로 판단")
        log_access(role, query, hits, True, blocked, mode)
        return result

    # 3겹: 인용 검증
    result["unknown_citations"] = _validate_citations(raw, hits)
    result["answer"] = raw
    log_access(role, query, hits, False, blocked, mode)
    return result
