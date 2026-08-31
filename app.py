"""데모 UI:  streamlit run app.py

데모 시나리오 (심사 3분):
 1. 신입사원 역할로 "연차 신청은 며칠 전에?" -> 출처 인용된 답변
 2. 역할 그대로 "과장 연봉 밴드는?"        -> 차단된 청크 수가 뜨면서 거부
 3. 인사팀으로 전환 후 같은 질문            -> 답변됨
 4. "회사 창립기념일은?"                   -> 근거 없음, 지어내지 않고 거부
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from answer import ask  # noqa: E402
from config import ROLE_CLEARANCE, ROLE_LABEL  # noqa: E402
from llm import NoLLMKey  # noqa: E402

st.set_page_config(page_title="사내 온보딩 어시스턴트", layout="wide")

with st.sidebar:
    st.header("접속 계정")
    role = st.selectbox("역할", list(ROLE_LABEL), format_func=lambda r: ROLE_LABEL[r])
    st.caption("열람 권한 등급: " + ", ".join(sorted(ROLE_CLEARANCE[role])))
    st.divider()
    st.caption("이 어시스턴트는 사내 규정 문서에 근거가 있는 내용만 답변합니다. "
               "근거가 없으면 답변하지 않습니다.")

st.title("사내 온보딩 어시스턴트")
st.caption("On-premise 환경 기반 RAG 프로토타입")

query = st.chat_input("사내 규정에 대해 질문하세요")
if query:
    st.chat_message("user").write(query)
    with st.chat_message("assistant"):
        try:
            with st.spinner("사내 문서 검색 중"):
                res = ask(query, role)
        except NoLLMKey as e:
            st.error(str(e))
            st.stop()

        if res["abstained"]:
            st.warning(res["answer"])
            st.caption(f"거부 사유: {res['reason']}")
        else:
            st.write(res["answer"])
            if res["unknown_citations"]:
                st.error("검색 결과에 없는 출처가 인용되었습니다: "
                         + ", ".join(res["unknown_citations"]))

        c1, c2, c3 = st.columns(3)
        c1.metric("최고 유사도", f"{res['max_score']:.3f}")
        c2.metric("권한으로 차단된 청크", res["blocked_chunks"])
        c3.metric("답변 여부", "거부" if res["abstained"] else "응답")

        with st.expander(f"근거 문서 {len(res['hits'])}건", expanded=not res["abstained"]):
            for h in res["hits"]:
                st.markdown(
                    f"**{h['rank']}. {h['title']} / {h['section']}**  \n"
                    f"`{h['doc_id']}` · {h['version']} · 시행일 {h['effective_date']} · "
                    f"등급 {h['clearance']} · 유사도 {h['score']:.3f}")
                st.text(h["text"][:400])
                st.divider()
