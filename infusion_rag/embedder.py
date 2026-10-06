"""向量化后端。

默认后端 `TfidfVectorizer` 只用标准库实现（中英文字符 n-gram TF-IDF，L2 归一化），
因此仓库开箱即用、完全离线、无需下载模型。

如果安装了 sentence-transformers，可以切到语义向量后端 `SentenceTransformerEmbedder`。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Protocol

from .config import EmbedderConfig
from .tokenizer import tokenize

SparseVector = dict[int, float]
DenseVector = list[float]


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> list:  # noqa: D102
        ...


# --------------------------------------------------------------------------- #
# 默认：TF-IDF（纯标准库）
# --------------------------------------------------------------------------- #
class TfidfVectorizer:
    """词表 + IDF 的稀疏 TF-IDF，向量已做 L2 归一化，点积即余弦相似度。"""

    name = "tfidf"

    def __init__(self, min_df: int = 1, sublinear_tf: bool = True) -> None:
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.terms: list[str] = []
        self.term_index: dict[str, int] = {}
        self.idf: list[float] = []
        self.dim = 0
        self._fitted = False

    # -- 拟合 -------------------------------------------------------------- #
    def fit(self, texts: list[str]) -> "TfidfVectorizer":
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(tokenize(text)))

        kept = [(term, n) for term, n in df.items() if n >= self.min_df]
        # 稳定排序：保证索引构建可复现
        kept.sort(key=lambda item: (-item[1], item[0]))

        self.terms = [term for term, _ in kept]
        self.term_index = {term: i for i, term in enumerate(self.terms)}
        total = max(len(texts), 1)
        self.idf = [math.log((1 + total) / (1 + n)) + 1.0 for _, n in kept]
        self.dim = len(self.terms)
        self._fitted = True
        return self

    # -- 编码 -------------------------------------------------------------- #
    def _weights(self, text: str) -> SparseVector:
        counts = Counter(tokenize(text))
        vec: SparseVector = {}
        for term, count in counts.items():
            idx = self.term_index.get(term)
            if idx is None:
                continue
            tf = 1.0 + math.log(count) if (self.sublinear_tf and count > 0) else float(count)
            vec[idx] = tf * self.idf[idx]
        return _l2_normalize_sparse(vec)

    def encode(self, texts: Iterable[str]) -> list[SparseVector]:
        if not self._fitted:
            raise RuntimeError("TfidfVectorizer 需要先 fit()")
        return [self._weights(t) for t in texts]

    # -- 序列化 ------------------------------------------------------------ #
    def state(self) -> dict:
        return {
            "kind": self.name,
            "terms": self.terms,
            "idf": self.idf,
            "min_df": self.min_df,
            "sublinear_tf": self.sublinear_tf,
        }

    @classmethod
    def from_state(cls, state: dict) -> "TfidfVectorizer":
        obj = cls(
            min_df=int(state.get("min_df", 1)),
            sublinear_tf=bool(state.get("sublinear_tf", True)),
        )
        obj.terms = list(state["terms"])
        obj.idf = [float(x) for x in state["idf"]]
        obj.term_index = {term: i for i, term in enumerate(obj.terms)}
        obj.dim = len(obj.terms)
        obj._fitted = True
        return obj


# --------------------------------------------------------------------------- #
# 可选：sentence-transformers 语义向量
# --------------------------------------------------------------------------- #
class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise RuntimeError(
                "未安装 sentence-transformers。请执行：\n"
                "    pip install -r requirements-optional.txt\n"
                "或者改用默认后端 --backend tfidf"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Iterable[str]) -> list[DenseVector]:
        vectors = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(x) for x in row] for row in vectors]

    def state(self) -> dict:
        return {"kind": self.name, "model_name": self.model_name}


# --------------------------------------------------------------------------- #
# 工厂 / 数学工具
# --------------------------------------------------------------------------- #
def create_embedder(cfg: EmbedderConfig):
    backend = (cfg.backend or "tfidf").lower()
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401  # type: ignore
        except ImportError:
            backend = "tfidf"
        else:
            backend = "sentence-transformers"

    if backend in {"tfidf", "lexical", "default"}:
        return TfidfVectorizer(min_df=cfg.min_df)
    if backend in {"sentence-transformers", "st", "dense"}:
        return SentenceTransformerEmbedder(cfg.model_name)
    raise ValueError(f"未知的向量后端：{cfg.backend}")


def _l2_normalize_sparse(vec: SparseVector) -> SparseVector:
    norm = math.sqrt(sum(w * w for w in vec.values()))
    if norm <= 0:
        return {}
    return {idx: w / norm for idx, w in vec.items()}


def normalize_dense(vec: DenseVector) -> DenseVector:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0:
        return vec
    return [x / norm for x in vec]


def dot_sparse(a: SparseVector, b: SparseVector) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(idx, 0.0) for idx, w in a.items())


def dot_dense(a: DenseVector, b: DenseVector) -> float:
    return sum(x * y for x, y in zip(a, b))
