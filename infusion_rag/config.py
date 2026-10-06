"""配置：路径、分块参数、向量后端、可选大模型。"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_INDEX_PATH = REPO_ROOT / "index" / "kb_index.json"

DEFAULT_EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"


@dataclass
class ChunkConfig:
    """分块参数。

    max_chars 是目标上限（由于会保留重叠，实际长度可能略超）。
    overlap_chars 是相邻块之间的重叠字符数，用于避免答案被切断。
    """

    max_chars: int = 600
    overlap_chars: int = 100
    min_chars: int = 80

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EmbedderConfig:
    """向量后端配置。

    backend = "tfidf"                仅用标准库实现的中英文字符 n-gram TF-IDF（默认，离线可用）
    backend = "sentence-transformers" 需要安装 sentence-transformers
    backend = "auto"                 装了 sentence-transformers 就用它，否则退回 tfidf
    """

    backend: str = "tfidf"
    model_name: str = DEFAULT_EMBEDDING_MODEL
    min_df: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LLMConfig:
    """可选的大模型配置（任何 OpenAI 兼容 /chat/completions 接口）。

    通过环境变量提供：
        RAG_LLM_BASE_URL   例如 https://api.deepseek.com/v1
        RAG_LLM_API_KEY    密钥
        RAG_LLM_MODEL      例如 deepseek-chat
    """

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.2
    timeout: int = 90
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.environ.get("RAG_LLM_BASE_URL", "").strip(),
            api_key=os.environ.get("RAG_LLM_API_KEY", "").strip(),
            model=os.environ.get("RAG_LLM_MODEL", "").strip(),
            temperature=float(os.environ.get("RAG_LLM_TEMPERATURE", "0.2") or 0.2),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["api_key"] = "***" if self.api_key else ""
        return d
