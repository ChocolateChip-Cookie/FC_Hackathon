"""임베딩 백엔드.

기본은 로컬 오픈 모델(BGE-M3)이다. 온프레미스가 기본 모드이므로 임베딩을 외부로 보내면
요건 자체가 무너진다. upstage / openai 는 클라우드 모드(더미데이터 체험용)이고,
hash 는 모델도 키도 없을 때 로직만 돌려보기 위한 오프라인 폴백이다.

백엔드 선택은 providers.embed_backend() 한 곳에서만 한다.

Upstage 는 OpenAI 호환이라 같은 HTTP 경로를 쓴다. 단 solar-embedding 은 **비대칭 모델**이다:
문서 색인은 -passage, 검색 질의는 -query 를 써야 한다. 뒤바꿔도 에러가 나지 않고 점수만
조용히 나빠지므로, 호출부에서 is_query 를 명시하게 만들었다.
"""
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request

import numpy as np

from config import EMBED_API, EMBED_MODEL_LOCAL
from providers import embed_backend

DIM_HASH = 2048
_model = None


def backend_name() -> str:
    return embed_backend()


# ---------- 로컬 오픈 모델 (기본) ----------
def _load_local():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL_LOCAL)
    return _model


def _local_embed(texts: list[str]) -> np.ndarray:
    return _load_local().encode(
        texts, normalize_embeddings=True, batch_size=16,
        show_progress_bar=False).astype(np.float32)


# ---------- 오프라인 폴백: 문자 3-gram 해싱 ----------
def _hash_vector(text: str) -> np.ndarray:
    vec = np.zeros(DIM_HASH, dtype=np.float32)
    t = "".join(text.split())
    grams = [t[i:i + 3] for i in range(max(len(t) - 2, 1))]
    counts: dict[int, int] = {}
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % DIM_HASH
        counts[h] = counts.get(h, 0) + 1
    for h, c in counts.items():
        vec[h] = 1.0 + math.log(c)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


# ---------- OpenAI 호환 API (openai / upstage 공용) ----------
def _api_embed(texts: list[str], backend: str, is_query: bool) -> np.ndarray:
    spec = EMBED_API[backend]
    key = os.environ.get(spec["key_env"], "")
    if not key:
        raise RuntimeError(f"{spec['key_env']} 가 설정되지 않았습니다 (.env.local 확인)")
    model = spec["query"] if is_query else spec["passage"]

    req = urllib.request.Request(
        f"{spec['base_url']}/embeddings",
        data=json.dumps({"model": model, "input": texts}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    last_err = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code != 429 or attempt == 5:
                raise
            wait = min(60, 2 ** (attempt + 1))
            print(f"embed 429, {wait}s 후 재시도 ({attempt + 1}/6)", flush=True)
            time.sleep(wait)
    else:
        raise last_err
    # 응답 순서가 입력 순서와 다를 수 있으므로 index 로 정렬한다.
    rows = sorted(body["data"], key=lambda d: d["index"])
    arr = np.array([d["embedding"] for d in rows], dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def embed(texts: list[str], is_query: bool = True) -> np.ndarray:
    """L2 정규화된 (n, dim) 행렬. 정규화했으므로 내적 = 코사인 유사도.

    is_query: 검색 질의면 True, 문서 색인이면 False.
              비대칭 모델(solar-embedding)에서만 실제로 갈리지만, 호출부가 자기 의도를
              밝히도록 기본값 대신 항상 넘기는 것을 권한다.
    """
    if not texts:
        return np.zeros((0, DIM_HASH), dtype=np.float32)
    backend = backend_name()
    if backend == "bge":
        return _local_embed(texts)
    if backend in EMBED_API:
        out = []
        for i in range(0, len(texts), 64):   # 배치
            out.append(_api_embed(texts[i:i + 64], backend, is_query))
        return np.vstack(out)
    if backend == "hash":
        return np.vstack([_hash_vector(t) for t in texts])
    raise RuntimeError(
        "임베딩 백엔드가 없습니다. BM25 검색만 사용하거나, .env.local 에 "
        "UPSTAGE_API_KEY 또는 OPENAI_API_KEY 를 넣으십시오.")
