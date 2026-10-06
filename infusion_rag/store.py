"""索引的持久化。

拆成两个文件：

    index/kb_index.json      BM25 统计量 + 片段 + 文档元数据（小、可读、可 diff）
    index/kb_index.dense.json  稠密向量（只在 --dense 构建时存在，体积大得多）

分开是因为稠密向量动辄几百 KB 到几 MB，混在一起会让主索引完全失去可读性，
而绝大多数使用场景（纯 BM25）根本不需要它。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .bm25 import BM25Index
from .chunker import Chunk
from .frontmatter import DocMeta

INDEX_VERSION = 3


@dataclass
class KnowledgeIndex:
    chunks: list[Chunk] = field(default_factory=list)
    docs: dict[str, DocMeta] = field(default_factory=dict)
    bm25: BM25Index = field(default_factory=BM25Index)
    dense_vectors: list[list[float]] | None = None
    dense_model: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def has_dense(self) -> bool:
        return bool(self.dense_vectors)

    @property
    def vocab_size(self) -> int:
        return len(self.bm25.terms)

    def __len__(self) -> int:
        return len(self.chunks)

    def doc(self, doc_id: str) -> DocMeta | None:
        return self.docs.get(doc_id)

    def sources_of(self, chunk: Chunk) -> list[dict]:
        meta = self.docs.get(chunk.doc_id)
        return [s.to_dict() for s in meta.sources] if meta else []

    # ------------------------------------------------------------------ #
    def to_payload(self) -> dict:
        return {
            "version": INDEX_VERSION,
            "meta": self.meta,
            "docs": {k: v.to_dict() for k, v in self.docs.items()},
            "chunks": [c.to_dict() for c in self.chunks],
            "bm25": self.bm25.state(),
        }

    @classmethod
    def from_payload(cls, payload: dict, dense: dict | None = None) -> "KnowledgeIndex":
        version = payload.get("version")
        if version != INDEX_VERSION:
            raise ValueError(
                f"索引格式版本不匹配（文件 {version}，当前程序 {INDEX_VERSION}）。"
                f"请重新运行：python -m infusion_rag.cli build"
            )
        dense_vectors = None
        dense_model = ""
        if dense and dense.get("vectors"):
            dense_vectors = [[float(x) for x in row] for row in dense["vectors"]]
            dense_model = str(dense.get("model_name", ""))

        return cls(
            chunks=[Chunk.from_dict(c) for c in payload["chunks"]],
            docs={k: DocMeta.from_dict(v) for k, v in payload["docs"].items()},
            bm25=BM25Index.from_state(payload["bm25"]),
            dense_vectors=dense_vectors,
            dense_model=dense_model,
            meta=payload.get("meta", {}),
        )

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.meta.setdefault(
            "built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        path.write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        dense_path = dense_path_for(path)
        if self.has_dense:
            dense_path.write_text(
                json.dumps(
                    {"model_name": self.dense_model, "vectors": self.dense_vectors},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
        elif dense_path.exists():
            dense_path.unlink()  # 这次没建稠密向量，别留下上次的旧文件
        return path

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeIndex":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"索引文件不存在：{path}\n请先运行：python -m infusion_rag.cli build"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        dense_path = dense_path_for(path)
        dense = json.loads(dense_path.read_text(encoding="utf-8")) if dense_path.exists() else None
        return cls.from_payload(payload, dense)


def dense_path_for(index_path: str | Path) -> Path:
    """稠密向量文件与主索引同目录，文件名加 .dense 后缀。"""
    path = Path(index_path)
    return path.with_name(f"{path.stem}.dense{path.suffix}")
