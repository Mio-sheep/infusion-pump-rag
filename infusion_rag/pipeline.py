"""把 分块 → 向量化 → 检索 → 生成 串成一条流水线。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .chunker import chunk_corpus
from .config import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_INDEX_PATH,
    ChunkConfig,
    EmbedderConfig,
    LLMConfig,
)
from .embedder import (
    SentenceTransformerEmbedder,
    TfidfVectorizer,
    create_embedder,
)
from .generator import extractive_answer, llm_answer
from .retriever import Hit, Retriever, term_coverage
from .store import KnowledgeIndex

__version__ = "1.0.0"


class InfusionPumpRAG:
    """小型 RAG 知识库（输液泵 / 注射泵）。"""

    def __init__(self, index: KnowledgeIndex, *, index_path: Path | None = None,
                 use_synonyms: bool = True) -> None:
        self.index = index
        self.index_path = Path(index_path) if index_path else None
        self.retriever = Retriever(index, use_synonyms=use_synonyms)

    # ------------------------------------------------------------------ #
    # 构建 / 加载
    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        corpus_dir: str | Path | None = None,
        index_path: str | Path | None = None,
        chunk_cfg: ChunkConfig | None = None,
        embedder_cfg: EmbedderConfig | None = None,
        verbose: bool = False,
    ) -> "InfusionPumpRAG":
        corpus_dir = Path(corpus_dir or DEFAULT_CORPUS_DIR)
        index_path = Path(index_path or DEFAULT_INDEX_PATH)
        chunk_cfg = chunk_cfg or ChunkConfig()
        embedder_cfg = embedder_cfg or EmbedderConfig()

        log = (lambda msg: print(msg)) if verbose else (lambda msg: None)

        log(f"[1/4] 读取语料：{corpus_dir}")
        chunks = chunk_corpus(corpus_dir, chunk_cfg)
        if not chunks:
            raise RuntimeError(f"语料目录里没有可用文档：{corpus_dir}")
        docs = sorted({c.doc_id for c in chunks})
        log(f"      {len(docs)} 篇文档 → {len(chunks)} 个片段")

        log("[2/4] 构建稀疏 TF-IDF 向量")
        texts = [c.text for c in chunks]
        vectorizer = TfidfVectorizer(min_df=embedder_cfg.min_df).fit(texts)
        sparse_vectors = vectorizer.encode(texts)
        log(f"      词表 {vectorizer.dim} 个 term")

        dense_vectors = None
        dense_model = ""
        backend = (embedder_cfg.backend or "tfidf").lower()
        if backend not in {"tfidf", "lexical", "default"}:
            log(f"[3/4] 构建稠密向量（backend={embedder_cfg.backend}）")
            try:
                embedder = create_embedder(embedder_cfg)
            except RuntimeError as exc:
                print(f"      ! {exc}\n      → 已自动回退为 tfidf 后端")
                embedder = None
            if isinstance(embedder, SentenceTransformerEmbedder):
                dense_vectors = embedder.encode(texts)
                dense_model = embedder.model_name
                log(f"      维度 {embedder.dim}，模型 {dense_model}")
        else:
            log("[3/4] 跳过稠密向量（backend=tfidf）")

        meta = {
            "name": "infusion-pump-rag",
            "version": __version__,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "corpus_dir": str(corpus_dir),
            "documents": docs,
            "n_documents": len(docs),
            "n_chunks": len(chunks),
            "vocab_size": vectorizer.dim,
            "dense_model": dense_model,
            "chunk_config": chunk_cfg.to_dict(),
        }

        index = KnowledgeIndex(
            chunks=chunks,
            sparse_vectors=sparse_vectors,
            tfidf_state=vectorizer.state(),
            dense_vectors=dense_vectors,
            dense_model=dense_model,
            meta=meta,
        )

        log(f"[4/4] 写入索引：{index_path}")
        index.save(index_path)
        log("      完成 ✓")
        return cls(index, index_path=index_path)

    @classmethod
    def load(cls, index_path: str | Path | None = None,
             use_synonyms: bool = True) -> "InfusionPumpRAG":
        path = Path(index_path or DEFAULT_INDEX_PATH)
        return cls(KnowledgeIndex.load(path), index_path=path, use_synonyms=use_synonyms)

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 4, mode: str = "hybrid") -> list[Hit]:
        return self.retriever.search(query, top_k=top_k, mode=mode)

    def ask(
        self,
        query: str,
        top_k: int = 4,
        mode: str = "hybrid",
        llm: LLMConfig | None = None,
        use_llm: bool | None = None,
    ) -> dict:
        """检索 + 生成。use_llm=None 表示"配了大模型就用"。"""
        hits = self.search(query, top_k=top_k, mode=mode)
        llm = llm if llm is not None else LLMConfig.from_env()
        if use_llm is None:
            use_llm = llm.enabled

        if use_llm and llm.enabled:
            answer = llm_answer(query, hits, llm)
            generator = "llm"
        else:
            answer = extractive_answer(query, hits)
            generator = "extractive"

        return {
            "query": query,
            "answer": answer,
            "generator": generator,
            "mode": mode,
            "n_hits": len(hits),
            "hits": [hit.to_dict() for hit in hits],
            "citations": [
                {
                    "rank": hit.rank,
                    "doc_title": hit.chunk.doc_title,
                    "heading": hit.chunk.heading,
                    "source": hit.chunk.source,
                    "chunk_id": hit.chunk.chunk_id,
                }
                for hit in hits
            ],
        }

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        meta = dict(self.index.meta)
        lengths = [len(v) for v in self.index.sparse_vectors] or [0]
        headings = {c.heading for c in self.index.chunks if c.heading}
        meta.update(
            {
                "index_path": str(self.index_path) if self.index_path else "",
                "n_chunks": len(self.index),
                "n_sections": len(headings),
                "avg_chunk_chars": round(
                    sum(len(c.text) for c in self.index.chunks) / max(len(self.index), 1), 1
                ),
                "avg_terms_per_chunk": round(sum(lengths) / max(len(lengths), 1), 1),
                "vocab_size": self.index.dim,
                "vocab_coverage": term_coverage(self.index),
                "has_dense": self.index.has_dense,
                "dense_model": self.index.dense_model,
            }
        )
        return meta

    def documents(self) -> list[dict]:
        grouped: dict[str, dict] = {}
        for chunk in self.index.chunks:
            row = grouped.setdefault(
                chunk.doc_id,
                {"doc_id": chunk.doc_id, "title": chunk.doc_title,
                 "source": chunk.source, "chunks": 0, "chars": 0},
            )
            row["chunks"] += 1
            row["chars"] += len(chunk.text)
        return sorted(grouped.values(), key=lambda r: r["source"])
