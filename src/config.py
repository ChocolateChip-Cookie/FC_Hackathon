"""전역 설정. 임계값은 반드시 eval/evaluate.py 로 튜닝한 뒤 확정할 것."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------- .env.local 로딩 ----------
# 내려받은 사람이 키를 넣는 자리다. 형식은 .env.local.example 참조.
# setdefault 인 이유: 실제 셸 환경변수가 파일보다 우선해야 한다. 파일이 셸을 덮으면
# 임시로 키를 바꿔 시험하는 것이 불가능해진다.
# 새 의존성(python-dotenv)을 쓰지 않는 이유: 이 리포는 llm.py 도 requests 대신 urllib 을
# 쓴다. 내려받는 쪽 설치 부담을 늘리지 않는다.
_ENV_FILE = ROOT / ".env.local"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip("\"'"))

POLICY_DIR = ROOT / "data" / "policies"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "policies"
# 제공자별 사전 생성 인덱스. 임베딩 공간이 다르면 색인과 질의가 어긋나므로
# 백엔드마다 파일이 따로 있어야 한다. 32청크라 파일당 130~500KB 수준이고 커밋한다.
INDEX_DIR = ROOT / "data" / "index"
LOG_PATH = ROOT / "data" / "access_log.jsonl"
GOLDEN_SET = ROOT / "eval" / "golden_set.json"

# 보안 모델: 사용자 역할 -> 열람 가능한 문서 등급
ROLE_CLEARANCE = {
    "intern": {"public"},                            # 신입/수습
    "employee": {"public", "internal"},              # 일반 직원
    "hr": {"public", "internal", "confidential"},    # 인사팀
}
ROLE_LABEL = {"intern": "신입사원", "employee": "일반 직원", "hr": "인사팀"}

TOP_K = 4

# ---------- 검색 방식 ----------
# dense    : 임베딩 코사인 유사도만
# bm25     : 형태소 토큰 BM25만
# ensemble : alpha*dense + (1-alpha)*bm25  (alpha 는 UI 에서 사용자가 조절)
RETRIEVAL_MODES = ("dense", "bm25", "ensemble")
DEFAULT_MODE = "ensemble"
# dense 쪽 가중치. 1.0 이면 dense, 0.0 이면 bm25 와 같아진다.
# 0.8 은 `python eval/evaluate.py --alpha-scan` 으로 **개발셋에서만** 고른 값이다.
# 기준은 MRR 이 아니라 '분리도'(답변 가능 문항 점수 > 거부 문항 점수인 쌍의 비율)다.
# MRR 은 모든 alpha 에서 1.000 이라 변별력이 없었다: 답변 가능 문항의 정답 문서가 항상
# 1순위였다는 뜻이고, 이 시스템의 병목이 검색 순위가 아니라 거부 판정이라는 증거다.
# 분리도는 0.0 에서 0.727, 0.8 에서 0.836 으로 오르고 0.8~1.0 이 동점이다.
# 동점 구간에서는 조항 번호·부서명 같은 정확한 어휘 일치를 위해 BM25 신호를 남기는
# 낮은 쪽을 택했다.
ENSEMBLE_ALPHA = 0.8

# BM25 원점수는 상한이 없어서 dense 코사인(0~1)과 그대로 더할 수 없다.
# 질의마다 min-max 정규화를 하면 최고점이 항상 1.0 이 되어 '거부 임계값'이 무력화되므로,
# 질의와 무관한 고정 포화 변환 raw/(raw+K) 로 0~1 에 넣는다. 스케일이 질의마다 흔들리지 않는다.
# ponytail: K 는 고정 상수. 문서가 늘어 점수 분포가 달라지면 재조정 대상이지만,
#           방식별 임계값을 스윕으로 따로 튜닝하므로 K 의 정확한 값 자체는 결과를 좌우하지 않는다.
BM25_SATURATION_K = 5.0

# 신뢰도 레이어: 최고 유사도가 이 값 미만이면 답변하지 않고 거부한다.
# 임베딩 백엔드와 검색 방식마다 점수 분포가 다르므로 (백엔드, 방식) 조합마다 값이 다르다.
# 이 값을 방식별로 따로 튜닝하지 않으면 3방식 비교가 무의미해진다
# (한 방식에만 유리한 임계값으로 나머지를 재는 꼴이 된다).
ABSTAIN_THRESHOLD = {
    # bge 값은 개발셋 스윕으로 확정 (`evaluate.py --compare`). alpha=0.8 기준.
    "bge":     {"dense": 0.56, "bm25": 0.47, "ensemble": 0.56},
    "upstage": {"dense": 0.50, "bm25": 0.47, "ensemble": 0.45},   # 미측정. 스윕으로 확정할 것
    "openai":  {"dense": 0.38, "bm25": 0.47, "ensemble": 0.34},   # 미측정. 스윕으로 확정할 것
    "hash":    {"dense": 0.11, "bm25": 0.47, "ensemble": 0.20},
    "none":    {"dense": 0.00, "bm25": 0.47, "ensemble": 0.00},   # 임베딩 키 없음. bm25 만 유효
}

# ---------- 모델 ----------
# 온프레미스가 기본 모드이므로 임베딩 기본값은 로컬 오픈 모델이다.
# 클라우드 모드(API 임베딩)는 더미데이터 체험용이며, 사내 실데이터에 쓰지 않는다.
EMBED_MODEL_LOCAL = "BAAI/bge-m3"          # 1024차원

# API 임베딩 백엔드. Upstage 는 OpenAI 호환이라 같은 코드 경로를 쓴다.
# solar-embedding 은 비대칭 모델이다: 색인은 -passage, 질의는 -query.
# 이걸 뒤바꾸면 점수가 조용히 나빠지고 원인을 찾기 어렵다.
EMBED_API = {
    "upstage": {
        "base_url": "https://api.upstage.ai/v1",
        "key_env": "UPSTAGE_API_KEY",
        "passage": "solar-embedding-1-large-passage",
        "query": "solar-embedding-1-large-query",
        "dim": 2048,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "passage": "text-embedding-3-small",
        "query": "text-embedding-3-small",   # 대칭 모델
        "dim": 1536,
    },
}

# 생성 백엔드. 우선순위는 providers.py 가 정한다 (온프렘이 먼저).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL_OLLAMA = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
CHAT_MODEL_ANTHROPIC = "claude-sonnet-4-5"
CHAT_MODEL_OPENAI = "gpt-4o-mini"
CHAT_MODEL_UPSTAGE = "solar-pro"

# ---------- 멀티쿼리 / 멀티턴 ----------
MULTI_QUERY_N = 3        # 원 질문 포함 총 질의 수
HISTORY_TURNS = 4        # 후속 질문 재작성에 참고할 직전 대화 턴 수
