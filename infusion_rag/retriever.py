"""检索。

三种模式：

    bm25    只用字段化 BM25。默认，零依赖，对术语和标准号这类查询最可靠。
    dense   只用稠密语义向量。需要先用 --dense 构建索引。
    hybrid  两路各自排序，再用 RRF 融合。措辞与关键词不匹配时明显更好。

混合为什么用 RRF（Reciprocal Rank Fusion）而不是加权求和：BM25 的分数是无界的，
余弦相似度是 [-1,1]，两者的量纲没法直接相加；归一化又会随查询分布漂移。
RRF 只看名次不看分数，不需要调参，是这个场景下更稳的选择。

    score(d) = Σ_r 1 / (k + rank_r(d))     k = 60

查询扩展：中文查询会按 synonyms.py 补上对应英文术语。语料是中英混排的，
问"阻塞报警"时把 occlusion 一起查，召回会好一截。
"""

from __future__ import annotations

from .chunker import Chunk
from .dense import DenseEncoder, cosine
from .store import KnowledgeIndex
from .synonyms import expand_query
from .tokenizer import tokenize_for_index

RRF_K = 60
DEFAULT_CANDIDATES = 40
MODES = ("bm25", "dense", "hybrid")


class Hit:
    """一条检索结果。"""

    __slots__ = ("chunk", "score", "bm25", "dense", "rank", "sources")

    def __init__(
        self,
        chunk: Chunk,
        score: float,
        *,
        bm25: float = 0.0,
        dense: float = 0.0,
        rank: int = 0,
        sources: list[dict] | None = None,
    ) -> None:
        self.chunk = chunk
        self.score = score
        self.bm25 = bm25
        self.dense = dense
        self.rank = rank
        self.sources = sources or []

    @property
    def label(self) -> str:
        return self.chunk.label

    def to_dict(self, with_text: bool = True) -> dict:
        data = {
            "rank": self.rank,
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "doc_title": self.chunk.doc_title,
            "heading": self.chunk.heading,
            "path": self.chunk.path,
            "score": round(self.score, 6),
            "bm25": round(self.bm25, 6),
            "dense": round(self.dense, 6),
            "sources": self.sources,
        }
        if with_text:
            data["text"] = self.chunk.text
        return data

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Hit {self.chunk.chunk_id} score={self.score:.4f}>"


class Retriever:
    def __init__(
        self,
        index: KnowledgeIndex,
        *,
        top_k: int = 4,
        use_synonyms: bool = True,
        rrf_k: int = RRF_K,
    ) -> None:
        self.index = index
        self.top_k = top_k
        self.use_synonyms = use_synonyms
        self.rrf_k = rrf_k
        self._encoder: DenseEncoder | None = None

    # ------------------------------------------------------------------ #
    def _encode_query(self, query: str) -> list[str]:
        tokens = tokenize_for_index(query)
        return expand_query(tokens, query) if self.use_synonyms else tokens

    def _dense_encoder(self) -> DenseEncoder:
        if self._encoder is None:
            if not self.index.dense_model:
                raise RuntimeError(
                    "索引里没有稠密向量。用 `build --dense` 重建，或改用 --mode bm25。"
                )
            self._encoder = DenseEncoder(self.index.dense_model)
        return self._encoder

    @staticmethod
    def _ranking(scores: list[float], limit: int, threshold: float = 0.0) -> list[int]:
        ranked = sorted(
            (i for i, s in enumerate(scores) if s > threshold),
            key=lambda i: (-scores[i], i),
        )
        return ranked[:limit]

    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: str = "bm25",
        candidates: int = DEFAULT_CANDIDATES,
    ) -> list[Hit]:
        query = (query or "").strip()
        if not query:
            return []
        mode = (mode or "bm25").lower()
        if mode not in MODES:
            raise ValueError(f"未知检索模式 {mode!r}，可选：{', '.join(MODES)}")

        top_k = top_k or self.top_k

        bm25_scores: list[float] = []
        dense_scores: list[float] = []

        if mode in {"bm25", "hybrid"}:
            bm25_scores = self.index.bm25.score(self._encode_query(query))
        if mode in {"dense", "hybrid"}:
            dense_scores = self._dense_cosine(query)

        bm25_rank = self._ranking(bm25_scores, candidates)
        dense_rank = self._ranking(dense_scores, candidates)

        if mode == "dense" and not dense_rank:
            if not self.index.has_dense:
                raise RuntimeError(
                    "索引里没有稠密向量。用 `build --dense` 重建，或改用 --mode bm25。"
                )
            return []

        if mode == "hybrid" and bm25_rank and dense_rank:
            fused: dict[int, float] = {}
            for position, idx in enumerate(bm25_rank):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + position + 1)
            for position, idx in enumerate(dense_rank):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + position + 1)
            ordered = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
            score_of = lambda i: fused[i]  # noqa: E731
        else:
            primary, scores = (bm25_rank, bm25_scores) if bm25_rank else (dense_rank, dense_scores)
            ordered = primary[:top_k]
            score_of = lambda i: scores[i] if scores else 0.0  # noqa: E731

        return [
            Hit(
                chunk=self.index.chunks[idx],
                score=score_of(idx),
                bm25=bm25_scores[idx] if bm25_scores else 0.0,
                dense=dense_scores[idx] if dense_scores else 0.0,
                rank=position,
                sources=self.index.sources_of(self.index.chunks[idx]),
            )
            for position, idx in enumerate(ordered, start=1)
        ]

    def _dense_cosine(self, query: str) -> list[float]:
        if not self.index.has_dense or not self.index.dense_vectors:
            return []
        query_vec = self._dense_encoder().encode([query])[0]
        return [cosine(query_vec, vec) for vec in self.index.dense_vectors]

    # ------------------------------------------------------------------ #
    def explain(self, query: str, mode: str = "bm25", top_k: int = 5) -> str:
        """调试用：打印名次与两路分数。"""
        hits = self.search(query, top_k=top_k, mode=mode)
        tokens = self._encode_query(query)
        lines = [
            f"query   : {query}",
            f"mode    : {mode}",
            f"tokens  : {' '.join(tokens[:60])}{' …' if len(tokens) > 60 else ''}",
            f"index   : {len(self.index)} 片段 / 词表 {self.index.vocab_size}",
            "",
        ]
        for hit in hits:
            lines.append(
                f"  {hit.rank:>2}. {hit.chunk.chunk_id:<24} "
                f"score={hit.score:.5f}  bm25={hit.bm25:.3f}  dense={hit.dense:.3f}  "
                f"{hit.label}"
            )
        return "\n".join(lines)
