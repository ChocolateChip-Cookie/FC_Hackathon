"""문서 적재: front matter 파싱 -> 조항 단위 청킹 -> 임베딩 -> ChromaDB 저장.

청킹을 고정 길이가 아니라 '## 제N조' 헤딩 단위로 하는 이유:
규정 문서는 조항이 의미 단위이고, 인용할 때 '문서명 + 조항'을 그대로 출처로 쓸 수 있다.

BM25 인덱스는 여기서 따로 만들지 않는다. 검색 시점에 chroma 에 적재된 것과 똑같은
코퍼스로 만들어야 두 방식이 같은 청크를 본다는 것이 보장되기 때문이다.
"""
import json
import re
import sys

import numpy as np

from config import INDEX_DIR, POLICY_DIR, ROOT
from embeddings import backend_name, embed
from store import build

FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)


def parse_doc(path):
    raw = path.read_text(encoding="utf-8")
    m = FRONT.match(raw)
    if not m:
        raise ValueError(f"front matter 없음: {path}")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[m.end():]


def _split_csv(value):
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def chunk(meta, body):
    """'## ' 헤딩 단위로 자른다. 헤딩 자체를 본문에 남겨 검색 신호로 쓴다."""
    chunks = []
    parts = re.split(r"\n(?=## )", body.strip())
    base = {
        "doc_id": meta["doc_id"],
        "title": meta["title"],
        "clearance": meta["clearance"],
        "version": meta.get("version", ""),
        "effective_date": meta.get("effective_date", ""),
        "owner": meta.get("owner", ""),
        "category_l1": meta.get("category_l1", ""),
        "category_l2": meta.get("category_l2", ""),
        "norm_rank": int(meta.get("norm_rank", 5)),
        "norm_type": meta.get("norm_type", ""),
        "access_scope": meta.get("access_scope", "all"),
        "access_depts": ", ".join(_split_csv(meta.get("access_depts", ""))),
        "access_positions": ", ".join(_split_csv(meta.get("access_positions", ""))),
        "status": meta.get("status", "active"),
        "superseded_by": meta.get("superseded_by", ""),
    }
    for part in parts:
        part = part.strip()
        if not part:
            continue
        heading = part.splitlines()[0].lstrip("# ").strip()
        item = dict(base)
        item["section"] = heading
        item["text"] = f"[{meta['title']}] {part}"
        chunks.append(item)
    return chunks


def build_index():
    all_chunks = []
    skipped = 0
    docs = sorted(POLICY_DIR.glob("*.md"))
    for path in docs:
        meta, body = parse_doc(path)
        if meta.get("searchable", "true").lower() == "false":
            skipped += 1
            continue
        all_chunks.extend(chunk(meta, body))

    # is_query=False: 문서 색인이다. solar-embedding 처럼 비대칭인 모델에서 -passage 를 쓴다.
    vectors = embed([c["text"] for c in all_chunks], is_query=False)
    backend = backend_name()
    n = build(all_chunks, vectors)

    # 제공자별 사전 생성 인덱스로도 남긴다. 내려받은 사람이 같은 키가 없어도
    # 커밋된 인덱스로 바로 검색할 수 있게 하기 위한 것이다.
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out = INDEX_DIR / f"{backend}.npz"
    np.savez_compressed(
        out,
        vectors=vectors,
        meta=np.array(json.dumps(all_chunks, ensure_ascii=False)),
        backend=np.array(backend),
    )
    print(f"문서 {len(docs)}건 중 {skipped}건 제외 -> 청크 {len(all_chunks)}개, "
          f"차원 {vectors.shape[1]}, 백엔드 {backend}, chroma 적재 {n}건")
    print(f"사전 생성 인덱스: {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f}KB)")
    return all_chunks


if __name__ == "__main__":
    sys.exit(0 if build_index() else 1)
