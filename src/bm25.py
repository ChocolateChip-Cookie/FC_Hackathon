"""BM25 어휘 검색.

한국어를 공백으로만 자르면 조사 때문에 '연차는'과 '연차가'가 다른 토큰이 되어
어휘 매칭이 거의 동작하지 않는다. kiwipiepy 형태소 분석으로 내용어만 남긴다.

dense 검색이 놓치는 것을 BM25 가 잡는다: 정확한 숫자, 사내 약어(KOS, HOP),
문서에만 등장하는 고유 표현. 반대로 BM25 는 표현이 다르면 못 찾는다.
그래서 앙상블이 의미를 갖는다.
"""
from kiwipiepy import Kiwi

from config import BM25_SATURATION_K

# 내용어만 남긴다: 명사, 수사, 숫자, 외국어/한자, 동사/형용사 어간, 어근
_KEEP = ("NNG", "NNP", "NNB", "NR", "NP", "SN", "SL", "SH", "VV", "VA", "XR")
_kiwi = None


def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def tokenize(text: str) -> list[str]:
    return [t.form for t in _get_kiwi().tokenize(text) if t.tag in _KEEP]


def build(corpus: list[str]):
    """청크 텍스트 목록에서 BM25 인덱스를 만든다."""
    from rank_bm25 import BM25Okapi
    return BM25Okapi([tokenize(c) for c in corpus])


def scores(bm25, query: str):
    """0~1 로 포화 변환한 점수 배열을 돌려준다.

    원점수는 상한이 없어 dense 코사인과 그대로 더할 수 없다. 질의마다 min-max 정규화를
    하면 최고점이 항상 1.0 이 되어 거부 임계값이 무력화되므로, 질의와 무관한 고정 변환
    raw/(raw+K) 를 쓴다. 단조 증가라 순위는 보존되고 스케일만 안정된다.
    """
    raw = bm25.get_scores(tokenize(query))
    return raw / (raw + BM25_SATURATION_K)
