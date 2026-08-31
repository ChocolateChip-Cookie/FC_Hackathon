"""ChromaDB 저장소.

청크 원문·메타데이터·dense 벡터의 단일 출처다. BM25 인덱스도 여기서 꺼낸 코퍼스로
만들기 때문에 두 검색 방식이 같은 청크 집합을 본다는 것이 구조적으로 보장된다.

1차원 등급 필터는 chroma where 로 내려보낼 수 있다. 지금은 등급×소속×직책이라
메타데이터가 리스트라 where 절로 표현할 수 없다. dense 는 전 청크 점수를 받은 뒤
허용 ID 만 남기고, bm25 는 같은 허용 집합으로 마스킹한다. 권한 없는 청크는
top-k 와 LLM 프롬프트에 오르지 않는다.
"""
import chromadb

from config import CHROMA_DIR, COLLECTION, INDEX_DIR

_client = None
_cache = None

META_KEYS = (
    "doc_id", "title", "clearance", "version", "effective_date", "owner", "section",
    "status", "access_scope", "access_depts", "access_positions", "norm_rank",
    "category_l1", "category_l2", "norm_type", "superseded_by",
)


def _chroma_meta(chunk):
    """chroma 는 list 메타데이터를 거절한다. 스칼라만 남긴다."""
    out = {}
    for k in META_KEYS:
        v = chunk.get(k, "")
        if v is None:
            v = ""
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        out[k] = v
    return out


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
    """컬렉션을 비우고 다시 채운다. 재색인은 항상 전체 교체다."""
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
        metadatas=[_chroma_meta(c) for c in chunks],
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


def dense_scores(qvec, allowed_ids, n_corpus):
    """전 청크 점수를 구한 뒤 허용 ID 만 남긴다. {id: 코사인 유사도}.

    chroma where 로 2차원 권한을 표현할 수 없어서, 점수 계산 후 허용 집합으로
    자른다. n_results 를 허용 수만큼만 받으면 대외비가 top-k 를 채워 열람 가능
    조항이 빠지므로, 코퍼스 전체를 받은 다음 필터한다.
    """
    if not allowed_ids or n_corpus <= 0:
        return {}
    r = collection().query(
        query_embeddings=[qvec.tolist()],
        n_results=n_corpus,
    )
    if not r["ids"] or not r["ids"][0]:
        return {}
    allowed = set(allowed_ids)
    # chroma 는 코사인 '거리'를 준다. 유사도 = 1 - 거리.
    return {i: 1.0 - d for i, d in zip(r["ids"][0], r["distances"][0])
            if i in allowed}
