"""输液泵 / 注射泵 小型 RAG 知识库。

零依赖核心：只用 Python 标准库即可完成 分块 → 向量化 → 检索 → 问答。
可选接入 sentence-transformers 做语义向量，可选接入任意 OpenAI 兼容大模型做生成。
"""

from .config import ChunkConfig, EmbedderConfig, LLMConfig
from .chunker import Chunk, chunk_corpus
from .retriever import Hit, Retriever
from .pipeline import InfusionPumpRAG

__version__ = "1.0.0"

__all__ = [
    "ChunkConfig",
    "EmbedderConfig",
    "LLMConfig",
    "Chunk",
    "chunk_corpus",
    "Hit",
    "Retriever",
    "InfusionPumpRAG",
    "__version__",
]
