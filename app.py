"""데모 UI:  streamlit run app.py

데모 시나리오 (심사 3분):
 1. 신입사원 역할로 "연차 신청은 며칠 전에?" -> 출처 인용된 답변
 2. 역할 그대로 "과장 연봉 밴드는?"        -> 차단된 청크 수가 뜨면서 거부
 3. 인사팀으로 전환 후 같은 질문            -> 답변됨
 4. "회사 창립기념일은?"                   -> 근거 없음, 지어내지 않고 거부

화면 규칙은 DESIGN.md 를 따른다. 핵심 두 가지:
- 액센트는 #0066cc 하나뿐이고, 상태색은 3px 좌측 바와 배지 밖으로 나가지 않는다.
- 그림자는 시스템 전체에서 출처 카드 하나에만 붙는다. 근거가 곧 제품이기 때문이다.
"""
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from config import (ABSTAIN_THRESHOLD, ENSEMBLE_ALPHA, ROLE_CLEARANCE,
                    ROLE_LABEL)  # noqa: E402
from llm import NoLLMKey  # noqa: E402
from providers import capabilities, mode_label  # noqa: E402
from retriever import log_access, search, should_abstain  # noqa: E402

st.set_page_config(page_title="사내 온보딩 어시스턴트", layout="wide")

# DESIGN.md: 액센트 1개 + 상태색 4개. 상태색은 좌측 3px 바와 배지에만 쓴다.
st.markdown("""
<style>
  :root {
    --accent: #0066cc; --ink: #1d1d1f; --muted: #7a7a7a;
    --parchment: #f5f5f7; --hairline: #e0e0e0;
    --st-grounded: #059669; --st-abstain: #B45309;
    --st-blocked: #1E1B4B; --st-error: #DC2626;
  }
  html, body, [class*="css"] { font-family: Pretendard, system-ui, -apple-system, sans-serif; }
  .stApp { color: var(--ink); }
  .demo-banner {
    border-left: 3px solid var(--st-abstain); background: var(--parchment);
    padding: 12px 16px; margin-bottom: 24px; font-size: 14px; line-height: 1.5;
  }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 9999px;
    font-size: 12px; font-weight: 600; border: 1px solid currentColor;
    margin-right: 6px;
  }
  .state-block { border-left: 3px solid var(--hairline); padding: 4px 0 4px 16px; margin: 8px 0 24px 0; }
  /* 출처 카드. 시스템 전체에서 유일한 그림자다.
     Streamlit 다크 테마는 글자를 밝게 뒤집어서, 양피지 배경 위에서는
     제목이 사라진다. 카드 안 색은 테마와 무관하게 고정한다. */
  .source-card {
    background: var(--parchment); color: var(--ink);
    border-radius: 18px; padding: 24px;
    box-shadow: rgba(0,0,0,0.22) 3px 5px 30px 0; margin: 16px 0 40px 0;
  }
  .source-card, .source-card * { color: var(--ink) !important; }
  .source-title { font-size: 17px; font-weight: 600; line-height: 1.47; }
  .source-meta { color: var(--muted) !important; font-size: 14px;
                 font-variant-numeric: tabular-nums; margin-top: 4px; }
  .source-body { margin-top: 12px; font-size: 17px; line-height: 1.47;
                 white-space: pre-wrap; }
  .num { font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)

cap = capabilities()

# ---------- 사이드바 ----------
with st.sidebar:
    st.header("접속 계정")
    role = st.selectbox("역할", list(ROLE_LABEL), format_func=lambda r: ROLE_LABEL[r])
    st.caption("열람 권한 등급: " + ", ".join(sorted(ROLE_CLEARANCE[role])))

    st.divider()
    st.header("검색 방식")

    # 능력 매트릭스가 허용하는 방식만 고를 수 있다.
    labels = {"dense": "임베딩 (dense)", "bm25": "어휘 (BM25)", "ensemble": "앙상블"}
    mode = st.radio("방식", cap["modes"], format_func=lambda m: labels[m],
                    index=len(cap["modes"]) - 1)

    # 비활성인 방식은 숨기지 않고 사유를 보여준다.
    # 왜 안 되는지를 말해주지 않으면 내려받은 사람이 고장으로 오해한다.
    for m in ("dense", "ensemble"):
        if m not in cap["modes"] and m in cap["reasons"]:
            st.caption(f"{labels[m]} 사용 불가: {cap['reasons'][m]}")

    alpha = ENSEMBLE_ALPHA
    if mode == "ensemble":
        alpha = st.slider("앙상블 가중치 (높을수록 임베딩 쪽)",
                          0.0, 1.0, ENSEMBLE_ALPHA, 0.1)
        st.caption(f"기본값 {ENSEMBLE_ALPHA}는 개발셋 분리도로 고른 값입니다 "
                   "(`evaluate.py --alpha-scan`). 홀드아웃은 쓰지 않았습니다.")

    st.divider()
    st.caption(mode_label(cap))
    if not cap["can_generate"]:
        st.caption(cap["reasons"].get("generation", ""))

# ---------- 본문 ----------
st.title("사내 온보딩 어시스턴트")
st.caption("On-premise 환경 기반 RAG 프로토타입")

st.markdown(
    '<div class="demo-banner"><b>데모 버전입니다.</b> 등장하는 사내 규정은 전부 '
    '<b>(가상) 주식회사 케이넥스</b>의 것으로, 시연을 위해 생성한 더미데이터입니다. '
    '실존 기업·인물과 무관합니다.</div>',
    unsafe_allow_html=True)

if not cap["can_generate"]:
    st.info("생성 백엔드가 없어 **답변을 만들지 않고 검색 결과만** 보여줍니다. "
            "검색·권한 필터·거부 판정은 그대로 동작합니다.")


def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="color:{color}">{text}</span>'


def render_sources(hits):
    """검색된 청크를 출처 카드 안에 넣는다. 본문을 카드 밖에 두면
    다크 테마에서 빈 흰 상자로 보인다."""
    st.markdown(f"**근거 문서 {len(hits)}건**")
    for h in hits:
        title = html.escape(f'{h["rank"]}. {h["title"]} / {h["section"]}')
        meta = html.escape(
            f'{h["doc_id"]} · {h["version"]} · 시행일 {h["effective_date"]} · '
            f'등급 {h["clearance"]} · 유사도 {h["score"]:.3f}')
        body = html.escape(h["text"][:400])
        st.markdown(
            f'<div class="source-card">'
            f'<div class="source-title">{title}</div>'
            f'<div class="source-meta">{meta}</div>'
            f'<div class="source-body">{body}</div>'
            f'</div>', unsafe_allow_html=True)


query = st.chat_input("사내 규정에 대해 질문하세요")
if query:
    st.chat_message("user").write(query)
    with st.chat_message("assistant"):
        with st.spinner("사내 문서 검색 중"):
            hits, max_score, blocked, backend = search(query, role, mode, alpha)

        thr = ABSTAIN_THRESHOLD[backend][mode]
        abstained = (not hits) or should_abstain(max_score, backend, mode)

        badges = []
        if blocked:
            badges.append(badge(f"권한 차단 {blocked}", "#1E1B4B"))
        if abstained:
            badges.append(badge("근거 없음", "#B45309"))
        else:
            badges.append(badge(f"출처 {len(hits)}건", "#059669"))

        if abstained:
            st.markdown(
                "".join(badges) +
                '<div class="state-block" style="border-left-color:#B45309">'
                "사내 문서에서 근거를 찾지 못했습니다. 담당 부서에 문의하세요."
                f'<div class="source-meta">검색 최고 유사도 '
                f'<span class="num">{max_score:.3f}</span> &lt; 임계값 '
                f'<span class="num">{thr:.2f}</span> 이므로 답변을 만들지 않았습니다.</div>'
                "</div>", unsafe_allow_html=True)
            log_access(role, query, hits, True, blocked, mode)
            if hits:
                render_sources(hits)

        elif not cap["can_generate"]:
            # 생성 백엔드가 없다. 지어내지 않고 검색 결과만 낸다.
            st.markdown("".join(badges), unsafe_allow_html=True)
            log_access(role, query, hits, False, blocked, mode)
            render_sources(hits)

        else:
            try:
                from answer import ask
                with st.spinner("답변 작성 중"):
                    res = ask(query, role, mode=mode, alpha=alpha)
            except NoLLMKey as e:
                st.error(str(e))
                st.stop()

            if res["unknown_citations"]:
                badges.append(badge("검증 실패", "#DC2626"))
            st.markdown("".join(badges), unsafe_allow_html=True)

            if res["abstained"]:
                st.markdown(
                    '<div class="state-block" style="border-left-color:#B45309">'
                    f'{res["answer"]}<div class="source-meta">{res["reason"]}</div></div>',
                    unsafe_allow_html=True)
            else:
                st.write(res["answer"])
                if res["unknown_citations"]:
                    st.markdown(
                        '<div class="state-block" style="border-left-color:#DC2626">'
                        "검색 결과에 없는 출처가 인용되었습니다: "
                        + ", ".join(res["unknown_citations"]) + "</div>",
                        unsafe_allow_html=True)
            render_sources(res["hits"])

    st.session_state["last_stats"] = {
        "max_score": max_score, "blocked": blocked, "mode": mode,
    }

stats = st.session_state.get("last_stats")
if stats:
    with st.sidebar:
        st.divider()
        st.metric("최고 유사도", f"{stats['max_score']:.3f}")
        st.metric("권한으로 차단된 청크", stats["blocked"])
        st.metric("검색 방식", stats["mode"])
