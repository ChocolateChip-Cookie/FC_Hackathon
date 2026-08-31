"""가진 키에 따라 무엇이 가능한지 판정한다.

내려받은 사람은 anthropic / openai / upstage 키를 전부 넣을 수도, 일부만 넣을 수도,
하나도 안 넣을 수도 있다. 이 모듈이 그 조합을 하나의 능력표로 바꾼다.

설계의 중심 사실 두 가지:
1. **Anthropic 에는 임베딩 API 가 없다.** 앤트로픽 키만 있으면 생성은 되지만 dense 검색은
   불가능하다. 이 경우 BM25 전용으로 떨어진다.
2. **BM25 는 키가 하나도 없어도 완전히 동작한다.** 그래서 키 0개인 사람도 권한 필터 전수
   검사와 BM25 검색 평가를 그대로 재현할 수 있다. 이 프로젝트의 보안 축과 측정 축이
   외부 서비스에 의존하지 않는다는 뜻이다.

UI 와 CLI 가 같은 함수를 쓴다. 판정이 두 군데로 갈라지면 화면과 평가가 다른 말을 하게 된다.
"""
import os

from config import CHROMA_DIR, EMBED_API, INDEX_DIR, OLLAMA_HOST


def _has(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def _ollama_up(timeout: float = 0.4) -> bool:
    """Ollama 가 실제로 떠 있는지 본다. 설치 여부가 아니라 응답 여부가 기준이다."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def embed_backend() -> str:
    """쓸 임베딩 백엔드를 고른다. 온프렘이 기본이므로 로컬 모델이 1순위다.

    EMBED_BACKEND 로 강제할 수 있다 (평가에서 백엔드를 고정할 때 쓴다).
    """
    forced = os.environ.get("EMBED_BACKEND", "").strip()
    if forced:
        return forced
    if _local_model_ready():
        return "bge"
    for name, spec in EMBED_API.items():
        if _has(spec["key_env"]):
            return name
    return "none"


def _local_model_ready() -> bool:
    """BGE-M3 를 쓸 수 있는지. 임포트만 확인하고 모델을 로드하지는 않는다.

    로드까지 하면 이 함수 한 번에 2.2GB 를 읽게 되어 UI 가 멈춘다.
    실제 가중치 유무는 첫 임베딩 시점에 드러난다.
    """
    if os.environ.get("DISABLE_LOCAL_MODEL"):
        return False
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def gen_backend() -> str:
    """생성 백엔드. 온프렘 모드가 기본이므로 Ollama 가 1순위다."""
    forced = os.environ.get("GEN_BACKEND", "").strip()
    if forced:
        return forced
    if _ollama_up():
        return "ollama"
    for name, env in (("anthropic", "ANTHROPIC_API_KEY"),
                      ("upstage", "UPSTAGE_API_KEY"),
                      ("openai", "OPENAI_API_KEY")):
        if _has(env):
            return name
    return "none"


def index_ready(backend: str) -> bool:
    """그 백엔드의 인덱스가 실제로 있는지. 없으면 dense 검색이 불가능하다."""
    if backend == "none":
        return False
    return (INDEX_DIR / f"{backend}.npz").exists() or CHROMA_DIR.exists()


def capabilities() -> dict:
    """UI 와 CLI 가 공유하는 단일 판정.

    reasons 는 사람이 읽는 문장이다. 방식이 비활성일 때 화면에 그대로 띄운다.
    "왜 안 되는지"를 말해주지 않으면 내려받은 사람이 고장으로 오해한다.
    """
    eb, gb = embed_backend(), gen_backend()
    has_index = index_ready(eb)
    dense_ok = eb != "none" and has_index

    reasons = {}
    if eb == "none":
        reasons["dense"] = "임베딩 키가 없습니다 (UPSTAGE_API_KEY 또는 OPENAI_API_KEY, 또는 로컬 BGE-M3)"
    elif not has_index:
        reasons["dense"] = f"{eb} 인덱스가 없습니다. python src/ingest.py 로 생성하세요"
    if not dense_ok:
        reasons["ensemble"] = reasons.get("dense", "dense 검색이 불가능합니다")
    if gb == "none":
        reasons["generation"] = "생성 키가 없습니다. 검색 결과만 표시합니다 (Ollama 또는 API 키 필요)"

    return {
        "embed_backend": eb,
        "gen_backend": gb,
        "modes": ["dense", "bm25", "ensemble"] if dense_ok else ["bm25"],
        "can_generate": gb != "none",
        "onprem": eb == "bge" and gb == "ollama",
        "reasons": reasons,
    }


def uses_external_api(cap: dict) -> bool:
    """사내 문서가 외부로 나가는 경로가 하나라도 열려 있는가.

    CLAUDE.md §2 가 구분하는 축이 바로 이것이다. 로컬 백엔드만 쓰면 키를 갖고 있어도
    외부 호출은 일어나지 않는다. '키가 있다'와 '외부로 보낸다'는 다른 말이다.
    """
    local = {"bge", "hash", "none"}
    return cap["embed_backend"] not in local or cap["gen_backend"] not in {"ollama", "none"}


def mode_label(cap: dict) -> str:
    """사이드바에 띄울 현재 모드 한 줄."""
    parts = [
        {"bge": "로컬 BGE-M3", "hash": "오프라인 해시", "none": "임베딩 없음"}.get(
            cap["embed_backend"], f"{cap['embed_backend']} API"),
        {"ollama": "Ollama", "none": "생성 없음"}.get(
            cap["gen_backend"], f"{cap['gen_backend']} API"),
    ]
    body = " + ".join(parts)
    if cap["onprem"]:
        return f"온프렘 모드 · {body} · 외부 API 호출 없음"
    if not uses_external_api(cap):
        return f"온프렘 모드(부분) · {body} · 외부 API 호출 없음"
    return f"클라우드 모드 · {body} · 더미데이터 전용"


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # 자체 점검: 키 조합 4가지가 각각 맞는 능력표를 내는지.
    # 프레임워크 없이 이 파일만 실행하면 된다.
    def _probe(env: dict) -> dict:
        saved = {k: os.environ.get(k) for k in
                 ("UPSTAGE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                  "EMBED_BACKEND", "GEN_BACKEND", "DISABLE_LOCAL_MODEL")}
        for k in saved:
            os.environ.pop(k, None)
        os.environ["DISABLE_LOCAL_MODEL"] = "1"   # 로컬 모델 유무와 무관하게 키만 본다
        os.environ.update(env)
        try:
            return capabilities()
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    ollama = _ollama_up()
    print(f"Ollama 응답: {ollama}\n")

    c = _probe({"ANTHROPIC_API_KEY": "x"})
    assert c["modes"] == ["bm25"], c
    assert c["can_generate"] is True, c
    print("앤트로픽만  ->", c["modes"], "| 생성", c["gen_backend"])

    c = _probe({})
    assert c["modes"] == ["bm25"], c
    assert c["can_generate"] is ollama, c
    print("키 0개      ->", c["modes"], "| 생성", c["gen_backend"])

    c = _probe({"UPSTAGE_API_KEY": "x"})
    assert c["embed_backend"] == "upstage", c
    print("업스테이지  ->", c["modes"], "| 임베딩", c["embed_backend"])

    c = _probe({"OPENAI_API_KEY": "x"})
    assert c["embed_backend"] == "openai", c
    print("오픈AI      ->", c["modes"], "| 임베딩", c["embed_backend"])

    print("\n현재 환경:", capabilities())
    print("모드:", mode_label(capabilities()))
    print("\n자체 점검 통과")
