"""검색기. 권한 필터를 '생성 단계'가 아니라 '검색 단계'에서 적용하는 것이 핵심이다.

권한 없는 청크를 일단 검색해서 LLM 프롬프트에 넣은 뒤 답변에서 가리는 방식은
프롬프트 인젝션 한 번에 원문이 새어나간다. 아예 후보에서 제외해야 한다.
dense 는 chroma 의 where 절로, bm25 는 같은 허용 집합으로 마스킹해서 두 경로 모두
동일한 권한 경계를 통과한다.

검색 방식 3종(dense / bm25 / ensemble)은 같은 청크 집합과 같은 권한 경계를 공유하므로
정답률 차이가 곧 검색 방식의 차이다.
"""
import json
from datetime import datetime, timezone

import numpy as np

import bm25 as bm25_mod
from config import (ABSTAIN_THRESHOLD, DEFAULT_MODE, ENSEMBLE_ALPHA, LOG_PATH,
                    TOP_K, resolve_user, visible)
from embeddings import backend_name, embed
from store import dense_scores, load

_bm25 = None


def _get_bm25():
    global _bm25
    if _bm25 is None:
        _, docs, _ = load()
        _bm25 = bm25_mod.build(docs)
    return _bm25


def reset_cache():
    """재색인 직후 호출한다. 인덱스가 바뀌었는데 BM25 가 옛 코퍼스를 물고 있으면 안 된다."""
    global _bm25
    _bm25 = None


def _is_live(meta):
    return meta.get("status", "active") == "active"


def search(query: str, role=None, mode: str = DEFAULT_MODE,
           alpha: float = ENSEMBLE_ALPHA, top_k: int = TOP_K, user=None):
    """(hits, max_score, blocked, backend) 를 돌려준다. 점수는 방식과 무관하게 0~1.

    role 은 골든셋 키(dev, hr). user 는 UI 의 {dept, position}. 둘 중 하나.
    """
    ids, docs, metas = load()
    account = user if user is not None else resolve_user(role)
    live_idx = [i for i, m in enumerate(metas) if _is_live(m)]
    allow_idx = [i for i in live_idx if visible(metas[i], account)]
    blocked = len(live_idx) - len(allow_idx)
    backend = backend_name()
    if not allow_idx:
        return [], 0.0, blocked, backend

    combined: dict[str, float] = {}
    allowed_ids = [ids[i] for i in allow_idx]

    if mode in ("dense", "ensemble"):
        qv = embed([query], is_query=True)[0]
        dense = dense_scores(qv, allowed_ids, len(ids))
    if mode in ("bm25", "ensemble"):
        raw = bm25_mod.scores(_get_bm25(), query)
        lex = {ids[i]: float(raw[i]) for i in allow_idx}

    if mode == "dense":
        combined = dense
    elif mode == "bm25":
        combined = lex
    elif mode == "ensemble":
        # 두 점수 모두 0~1 고정 스케일이라 그대로 가중합할 수 있다.
        # 한쪽에만 있는 후보는 다른 쪽 점수를 0 으로 본다.
        for k in set(dense) | set(lex):
            combined[k] = alpha * dense.get(k, 0.0) + (1 - alpha) * lex.get(k, 0.0)
    else:
        raise ValueError(f"알 수 없는 검색 방식: {mode}")

    if not combined:
        return [], 0.0, blocked, backend

    pos = {cid: i for i, cid in enumerate(ids)}
    order = sorted(combined, key=lambda k: -combined[k])[:top_k]
    hits = []
    for rank, cid in enumerate(order, 1):
        i = pos[cid]
        h = dict(metas[i])
        h["text"] = docs[i]
        h["score"] = float(combined[cid])
        h["rank"] = rank
        h["chunk_id"] = cid
        hits.append(h)
    return hits, float(max(combined.values())), blocked, backend


def search_multi(queries: list[str], role=None, mode: str = DEFAULT_MODE,
                 alpha: float = ENSEMBLE_ALPHA, top_k: int = TOP_K, user=None):
    """멀티쿼리 검색. 여러 변형 질의의 결과를 청크별 최고점으로 합친다.

    합집합을 rank 로 융합(RRF)하지 않고 최고점을 쓰는 이유는 거부 레이어 때문이다.
    RRF 점수는 후보가 얼마나 잘 맞았는지와 무관하게 1위면 늘 같은 값이라, 아무것도
    맞지 않은 질의에서도 최고점이 높게 나와 임계값 거부가 무력화된다.
    """
    if len(queries) == 1:
        return search(queries[0], role, mode, alpha, top_k, user=user)

    best: dict[str, dict] = {}
    blocked = backend = None
    for q in queries:
        hits, _, blocked, backend = search(q, role, mode, alpha, top_k, user=user)
        for h in hits:
            prev = best.get(h["chunk_id"])
            if prev is None or h["score"] > prev["score"]:
                best[h["chunk_id"]] = h
    if not best:
        return [], 0.0, blocked or 0, backend or backend_name()

    ranked = sorted(best.values(), key=lambda h: -h["score"])[:top_k]
    for rank, h in enumerate(ranked, 1):
        h["rank"] = rank
    return ranked, float(ranked[0]["score"]), blocked, backend


def should_abstain(max_score: float, backend: str, mode: str = DEFAULT_MODE) -> bool:
    return max_score < ABSTAIN_THRESHOLD[backend][mode]


def log_access(role, query, hits, abstained, blocked, mode=DEFAULT_MODE):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": role if isinstance(role, str) else resolve_user(role)["label"],
        "query": query,
        "mode": mode,
        "abstained": abstained,
        "blocked_chunks": blocked,
        "cited": [{"doc_id": h["doc_id"], "section": h["section"],
                   "score": round(h["score"], 4)} for h in hits],
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec
