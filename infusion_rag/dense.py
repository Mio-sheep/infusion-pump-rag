"""可选的稠密向量后端。

默认检索链路完全不需要这个模块：BM25 已经能覆盖关键词与术语检索。
只有当查询用的是完全不同的措辞时（例如问"泵一直响个不停"而不是"报警"），
语义向量才有明显价值。因此这里做成可选依赖，装不上就自动跳过。

    pip install -r requirements-optional.txt
    python -m infusion_rag.cli build --dense
"""
from __future__ import annotations

import math

DEFAULT_MODEL = "shibing624/text2vec-base-chinese"

INSTALL_HINT = (
    "未安装 sentence-transformers。安装方式：\n"
    "    pip install -r requirements-optional.txt\n"
    "或改用纯 BM25 检索（不加 --dense 即可）。"
)


class DenseEncoder:
    """sentence-transformers 的薄封装，输出已归一化的向量。"""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise RuntimeError(INSTALL_HINT) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        rows = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )
        return [[float(x) for x in row] for row in rows]

    # 稠密向量单独存一个文件，索引体积和加载速度都更可控
    def state(self) -> dict:
        return {"model_name": self.model_name, "dim": self.dim}


def cosine(a: list[float], b: list[float]) -> float:
    """两侧都已归一化时就是点积；这里仍做一次保护。"""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401  # type: ignore
    except ImportError:
        return False
    return True
