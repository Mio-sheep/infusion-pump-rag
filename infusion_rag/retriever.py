"""检索：稀疏 TF-IDF 余弦 + 可选稠密语义向量，用 RRF 融合。"""

from __future__ import annotations

import math

from .chunker import Chunk
from .embedder import (
    DenseVector,
    SentenceTransformerEmbedder,
    TfidfVectorizer,
    dot_dense,
    dot_sparse,
)
from .store import KnowledgeIndex
from .synonyms import expand_tokens
from .tokenizer import tokenize, tokenize_unique

RRF_K = 60

# 标题命中加成：查询词出现在小节标题里时给一点额外权重，
# 让"XX 是什么标准"这类问题优先命中真正讲该主题的小节，而不是参考来源列表。
HEADING_BOOST = 0.35

# 引用列表降权：文档末尾的"参考来源"小节里全是网址，关键词密度很高，
# 但那是出处清单而不是知识正文，按链接行占比做一次降权。
REFERENCE_PENALTY = 0.6
REFERENCE_LINK_RATIO = 0.3


class Hit:
    __slots__ = ("chunk", "score", "lexical", "dense", "rank")

    def __init__(self, chunk: Chunk, score: float, lexical: float = 0.0,
                 dense: float = 0.0, rank: int = 0) -> None:
        self.chunk = chunk
        self.score = score
        self.lexical = lexical
        self.dense = dense
        self.rank = rank

    @property
    def citation(self) -> str:
        head = self.chunk.heading or self.chunk.doc_title
        return f"[{self.rank}] {self.chunk.doc_title} › {head}"

    def to_dict(self, with_text: bool = True) -> dict:
        data = {
            "rank": self.rank,
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "doc_title": self.chunk.doc_title,
            "source": self.chunk.source,
            "heading": self.chunk.heading,
            "score": round(self.score, 6),
            "lexical": round(self.lexical, 6),
            "dense": round(self.dense, 6),
        }
        if with_text:
            data["text"] = self.chunk.text
        return data

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Hit {self.chunk.chunk_id} score={self.score:.4f}>"


class Retriever:
    """在 KnowledgeIndex 上做检索。

    mode:
        "lexical" 只用稀疏 TF-IDF
        "dense"   只用稠密语义向量（索引里没有稠密向量时会抛错）
        "hybrid"  两者都用（默认；没有稠密向量时自动退化为 lexical）
    """

    def __init__(self, index: KnowledgeIndex, *, top_k: int = 4,
                 use_synonyms: bool = True) -> None:
        self.index = index
        self.top_k = top_k
        self.use_synonyms = use_synonyms
        self.vectorizer = TfidfVectorizer.from_state(index.tfidf_state)
        self._dense_embedder: SentenceTransformerEmbedder | None = None
        self._penalties = [self._link_penalty(chunk.text) for chunk in index.chunks]

    # -- 降权 -------------------------------------------------------------- #
    @staticmethod
    def _link_penalty(text: str) -> float:
        """链接行占比越高，越可能是"参考来源"清单，给越低的分。"""
        lines = [line for line in text.split("\n") if line.strip()]
        if not lines:
            return 1.0
        links = sum(1 for line in lines if "http://" in line or "https://" in line)
        ratio = links / len(lines)
        weight = min(ratio / REFERENCE_LINK_RATIO, 1.0)
        return 1.0 - (1.0 - REFERENCE_PENALTY) * weight

    # -- 查询编码 ---------------------------------------------------------- #
    def _expand(self, query: str) -> str:
        if not self.use_synonyms:
            return query
        tokens = expand_tokens(tokenize_unique(query), query)
        return query + " " + " ".join(tokens)

    def _dense_encoder(self) -> SentenceTransformerEmbedder:
        if self._dense_embedder is None:
            model = self.index.dense_model or ""
            if not model:
                raise RuntimeError("索引里没有稠密向量，无法使用 dense 检索。")
            self._dense_embedder = SentenceTransformerEmbedder(model)
        return self._dense_embedder

    # -- 打分 -------------------------------------------------------------- #
    def _heading_boost(self, query: str) -> dict[int, float]:
        """按查询词与小节标题的词重叠给加成，只用长度 >= 2 的 token 以免单字噪声。"""
        query_terms = {t for t in tokenize_unique(query) if len(t) >= 2}
        if not query_terms:
            return {}
        boost: dict[int, float] = {}
        for i, chunk in enumerate(self.index.chunks):
            if not chunk.heading:
                continue
            heading_terms = {t for t in tokenize(chunk.heading) if len(t) >= 2}
            if not heading_terms:
                continue
            overlap = len(query_terms & heading_terms) / len(query_terms)
            if overlap:
                boost[i] = overlap
        return boost

    def _score_lexical(self, query: str) -> dict[int, float]:
        query_vec = self.vectorizer.encode([self._expand(query)])[0]
        if not query_vec:
            return {}
        scores = {
            i: dot_sparse(query_vec, vec) * self._penalties[i]
            for i, vec in enumerate(self.index.sparse_vectors)
            if vec
        }
        for idx, bonus in self._heading_boost(query).items():
            if idx in scores:
                scores[idx] += HEADING_BOOST * bonus
        return scores

    def _score_dense(self, query: str) -> dict[int, float]:
        if not self.index.has_dense or self.index.dense_vectors is None:
            return {}
        query_vec: DenseVector = self._dense_encoder().encode([query])[0]
        return {
            i: dot_dense(query_vec, vec)
            for i, vec in enumerate(self.index.dense_vectors)
            if vec
        }

    @staticmethod
    def _rank(scores: dict[int, float], limit: int) -> list[int]:
        return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]

    # -- 对外接口 ---------------------------------------------------------- #
    def search(self, query: str, top_k: int | None = None, mode: str = "hybrid",
               candidate_pool: int = 40) -> list[Hit]:
        query = (query or "").strip()
        if not query:
            return []

        top_k = top_k or self.top_k
        mode = (mode or "hybrid").lower()
        if mode not in {"lexical", "dense", "hybrid"}:
            raise ValueError(f"未知检索模式：{mode}")

        lexical = self._score_lexical(query) if mode in {"lexical", "hybrid"} else {}
        dense = self._score_dense(query) if mode in {"dense", "hybrid"} else {}

        if mode == "dense" and not dense:
            raise RuntimeError("索引中没有稠密向量，请用 --backend sentence-transformers 重新构建。")

        lex_rank = self._rank(lexical, candidate_pool)
        dense_rank = self._rank(dense, candidate_pool)

        if lex_rank and dense_rank:
            fused: dict[int, float] = {}
            for position, idx in enumerate(lex_rank):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + position + 1)
            for position, idx in enumerate(dense_rank):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + position + 1)
            ordered = self._rank(fused, top_k)
            score_of = lambda i: fused.get(i, 0.0)  # noqa: E731
        else:
            primary = lex_rank if lex_rank else dense_rank
            primary_scores = lexical if lex_rank else dense
            ordered = primary[:top_k]
            score_of = lambda i: primary_scores.get(i, 0.0)  # noqa: E731

        hits: list[Hit] = []
        for position, idx in enumerate(ordered, start=1):
            hits.append(
                Hit(
                    chunk=self.index.chunks[idx],
                    score=score_of(idx),
                    lexical=lexical.get(idx, 0.0),
                    dense=dense.get(idx, 0.0),
                    rank=position,
                )
            )
        return hits

    # -- 统计 -------------------------------------------------------------- #
    def explain(self, query: str, mode: str = "hybrid", top_k: int = 4) -> str:
        hits = self.search(query, top_k=top_k, mode=mode)
        lines = [f"query = {query!r}  mode = {mode}  命中 {len(hits)} 条"]
        for hit in hits:
            lines.append(
                f"  {hit.rank}. {hit.chunk.chunk_id}  "
                f"score={hit.score:.4f} (lex={hit.lexical:.4f}, dense={hit.dense:.4f})  "
                f"{hit.chunk.doc_title} › {hit.chunk.heading}"
            )
        return "\n".join(lines)


def cosine_sparse_stats(index: KnowledgeIndex) -> dict:
    """给 CLI 用的粗略统计。"""
    lengths = [len(v) for v in index.sparse_vectors]
    if not lengths:
        return {"chunks": 0}
    return {
        "chunks": len(lengths),
        "avg_terms_per_chunk": round(sum(lengths) / len(lengths), 1),
        "max_terms_per_chunk": max(lengths),
        "vocab": index.dim,
    }


def term_coverage(index: KnowledgeIndex) -> float:
    """被至少一个块用到的词表比例（大致反映词表裁剪情况）。"""
    if not index.dim:
        return 0.0
    used = set()
    for vec in index.sparse_vectors:
        used.update(vec.keys())
    return round(len(used) / index.dim, 3)


def _unused(*_args) -> None:  # pragma: no cover - 占位，避免 lint 误报
    _ = math
