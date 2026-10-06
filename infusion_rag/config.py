"""配置。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_INDEX_PATH = REPO_ROOT / "index" / "kb_index.json"
DEFAULT_EVAL_PATH = REPO_ROOT / "eval" / "questions.jsonl"


@dataclass
class ChunkConfig:
    """分块参数。

    max_chars 是目标上限；因为保留了重叠，实际片段长度可能略超这个值。
    """

    max_chars: int = 700
    overlap_chars: int = 120
    min_chars: int = 100

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BM25Config:
    """BM25 参数。

    k1   词频饱和速度：越大越看重"这个词出现了很多次"。
    b    长度归一化强度：0 表示不归一化，1 表示完全归一化。
    heading_weight  标题字段相对于正文字段的权重。
    """

    k1: float = 1.2
    b: float = 0.75
    heading_weight: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LLMConfig:
    """可选的大模型配置（任何 OpenAI 兼容的 /chat/completions 接口）。

    用环境变量提供：

        RAG_LLM_BASE_URL   例如 https://api.deepseek.com/v1
        RAG_LLM_API_KEY    密钥
        RAG_LLM_MODEL      例如 deepseek-chat
    """

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.2
    timeout: int = 90
    max_tokens: int = 1200

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            base_url=os.environ.get("RAG_LLM_BASE_URL", "").strip(),
            api_key=os.environ.get("RAG_LLM_API_KEY", "").strip(),
            model=os.environ.get("RAG_LLM_MODEL", "").strip(),
            temperature=float(os.environ.get("RAG_LLM_TEMPERATURE") or 0.2),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def to_dict(self) -> dict:
        return {
            "base_url": self.base_url,
            "api_key": "***" if self.api_key else "",
            "model": self.model,
            "temperature": self.temperature,
            "enabled": self.enabled,
        }
