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
            from sentence_transformers import SentenceTransformer
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


# --------------------------------------------------------------------------- #
# 可选：走 embedding API（默认硅基流动，OpenAI 兼容接口）
# --------------------------------------------------------------------------- #
class SiliconFlowEmbedder:
    """不下载本地模型，直接调 embedding API。

    什么时候选它：
      - 机器跑不动本地模型；
      - 或者只是想快速验证语义检索值不值得上。

    密钥从环境变量 SILICONFLOW_API_KEY 读取——**不要写进代码**，
    这个仓库是公开的，key 进了仓库就得全平台重置。
    """

    name = "siliconflow"
    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 16) -> None:
        import os

        api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "未设置环境变量 SILICONFLOW_API_KEY。"
                "到 https://cloud.siliconflow.cn 申请后设置：\n"
                '    [Environment]::SetEnvironmentVariable("SILICONFLOW_API_KEY", "sk-...", "User")'
            )

        from openai import OpenAI

        self.model_name = model_name
        self.batch_size = batch_size
        self.dim = 0  # 第一次真实调用后才知道，先占位
        self._client = OpenAI(api_key=api_key, base_url=self.DEFAULT_BASE_URL)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self._client.embeddings.create(input=batch, model=self.model_name)
            vectors.extend([float(x) for x in item.embedding] for item in resp.data)
        if not self.dim and vectors:
            self.dim = len(vectors[0])
        return vectors

    def state(self) -> dict:
        return {"model_name": self.model_name, "provider": self.name}


def create_dense_encoder(provider: str, model_name: str = DEFAULT_MODEL):
    """按 provider 建对应的稠密向量后端。

    provider = "local"        本地 sentence-transformers 模型
    provider = "siliconflow"  硅基流动 API
    """
    provider = (provider or "local").lower()
    if provider == "siliconflow":
        return SiliconFlowEmbedder(model_name)
    if provider == "local":
        return DenseEncoder(model_name)
    raise ValueError(f"未知的稠密向量提供方：{provider}")
