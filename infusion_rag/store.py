"""索引的持久化：一个 JSON 文件装下 分块 + 词表/IDF + 稀疏向量 + 可选的稠密向量。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .chunker import Chunk
from .embedder import DenseVector, SparseVector

INDEX_VERSION = 1


@dataclass
class KnowledgeIndex:
    chunks: list[Chunk] = field(default_factory=list)
    sparse_vectors: list[SparseVector] = field(default_factory=list)
    tfidf_state: dict = field(default_factory=dict)
    dense_vectors: list[DenseVector] | None = None
    dense_model: str = ""
    meta: dict = field(default_factory=dict)

    # -- 便捷属性 ---------------------------------------------------------- #
    @property
    def dim(self) -> int:
        return len(self.tfidf_state.get("terms", []))

    @property
    def has_dense(self) -> bool:
        return bool(self.dense_vectors)

    def __len__(self) -> int:
        return len(self.chunks)

    # -- 序列化 ------------------------------------------------------------ #
    def to_payload(self) -> dict:
        return {
            "version": INDEX_VERSION,
            "meta": self.meta,
            "tfidf": self.tfidf_state,
            "dense_model": self.dense_model,
            "chunks": [
                {
                    **chunk.to_dict(),
                    "sparse": {str(idx): round(w, 6) for idx, w in vector.items()},
                    **(
                        {"dense": [round(x, 6) for x in self.dense_vectors[i]]}
                        if self.has_dense and self.dense_vectors is not None
                        else {}
                    ),
                }
                for i, (chunk, vector) in enumerate(zip(self.chunks, self.sparse_vectors))
            ],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "KnowledgeIndex":
        version = payload.get("version")
        if version != INDEX_VERSION:
            raise ValueError(
                f"索引版本不匹配（文件 {version}，程序 {INDEX_VERSION}）。请重新运行 build。"
            )

        chunks: list[Chunk] = []
        sparse_vectors: list[SparseVector] = []
        dense_vectors: list[DenseVector] | None = [] if payload.get("dense_model") else None

        for row in payload.get("chunks", []):
            row = dict(row)
            sparse = {int(k): float(v) for k, v in (row.pop("sparse", {}) or {}).items()}
            dense = row.pop("dense", None)
            chunks.append(Chunk.from_dict(row))
            sparse_vectors.append(sparse)
            if dense_vectors is not None:
                dense_vectors.append([float(x) for x in (dense or [])])

        if dense_vectors is not None and not all(dense_vectors):
            dense_vectors = None

        return cls(
            chunks=chunks,
            sparse_vectors=sparse_vectors,
            tfidf_state=payload.get("tfidf", {}),
            dense_vectors=dense_vectors,
            dense_model=payload.get("dense_model", ""),
            meta=payload.get("meta", {}),
        )

    # -- 读写 -------------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.meta.setdefault("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        text = json.dumps(self.to_payload(), ensure_ascii=False, separators=(",", ":"))
        path.write_text(text, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeIndex":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"索引文件不存在：{path}\n请先运行：python -m infusion_rag.cli build"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_payload(payload)
