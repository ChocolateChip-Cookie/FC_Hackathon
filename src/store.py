"""ChromaDB 저장소.

청크 원문·메타데이터·dense 벡터의 단일 출처다. BM25 인덱스도 여기서 꺼낸 코퍼스로
만들기 때문에 두 검색 방식이 같은 청크 집합을 본다는 것이 구조적으로 보장된다.

권한 필터를 chroma 의 where 절로 내려보내는 것이 핵심이다. 애플리케이션에서 걸러내는
것이 아니라 DB 질의 조건 자체에 들어가므로, 권한 없는 청크는 후보 목록에 아예 오르지
않는다.
"""
import chromadb

from config import CHROMA_DIR, COLLECTION, INDEX_DIR

_client = None
_cache = None

META_KEYS = ("doc_id", "title", "clearance", "version", "effective_date", "owner", "section")


def _get_client():
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def collection():
    return _get_client().get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})


def build(chunks, vectors):
    """컬렉션을 비우고 다시 채운다. 재색인은 항상 전체 교체다 (32청크 규모)."""
    global _cache
    _cache = None
    client = _get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # 없으면 지울 것도 없다
    col = collection()
    col.add(
        ids=[f"{c['doc_id']}#{i}" for i, c in enumerate(chunks)],
        embeddings=[v.tolist() for v in vectors],
        documents=[c["text"] for c in chunks],
        metadatas=[{k: c.get(k, "") for k in META_KEYS} for c in chunks],
    )
    return col.count()


def restore_from_prebuilt(backend: str) -> int:
    """커밋된 사전 생성 인덱스를 chroma 에 적재한다.

    갓 클론한 사람에게 필요한 경로다. data/chroma/ 는 .gitignore 이므로 클론 직후에는
    비어 있고, 임베딩 키가 없으면 재생성도 못 한다. 커밋된 npz 가 그 간극을 메운다.
    """
    import json

    import numpy as np

    path = INDEX_DIR / f"{backend}.npz"
    if not path.exists():
        return 0
    z = np.load(path, allow_pickle=False)
    chunks = json.loads(str(z["meta"]))
    return build(chunks, z["vectors"])


def load(backend: str | None = None):
    """(ids, docs, metas) 를 고정된 순서로 돌려준다. BM25 코퍼스가 이 순서를 따른다.

    컬렉션이 비어 있으면 사전 생성 인덱스에서 복원을 한 번 시도한다.
    """
    global _cache
    if _cache is None:
        col = collection()
        if col.count() == 0:
            if backend is None:
                from providers import embed_backend
                backend = embed_backend()
            restore_from_prebuilt(backend)
            col = collection()
        g = col.get(include=["documents", "metadatas"])
        _cache = (list(g["ids"]), list(g["documents"]), list(g["metadatas"]))
    return _cache


def dense_scores(qvec, allowed, n):
    """권한 필터를 where 절로 내려 dense 유사도를 구한다. {id: 코사인 유사도}."""
    r = collection().query(
        query_embeddings=[qvec.tolist()],
        n_results=n,
        where={"clearance": {"$in": sorted(allowed)}},
    )
    if not r["ids"] or not r["ids"][0]:
        return {}
    # chroma 는 코사인 '거리'를 준다. 유사도 = 1 - 거리.
    return {i: 1.0 - d for i, d in zip(r["ids"][0], r["distances"][0])}
