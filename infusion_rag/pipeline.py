"""把 分块 → 建索引 → 检索 → 生成 串成一条流水线。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .bm25 import BM25Index
from .chunker import Document, load_corpus
from .config import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_EVAL_PATH,
    DEFAULT_INDEX_PATH,
    BM25Config,
    ChunkConfig,
    LLMConfig,
)
from .dense import DEFAULT_MODEL, DenseEncoder
from .eval import EvalReport, evaluate, load_questions
from .generator import extractive_answer, llm_answer
from .retriever import MODES, Hit, Retriever
from .store import KnowledgeIndex
from .tokenizer import tokenize_for_index

__version__ = "1.1.0"


class InfusionPumpRAG:
    """输液泵 / 注射泵 小型 RAG 知识库。"""

    def __init__(
        self,
        index: KnowledgeIndex,
        *,
        index_path: Path | None = None,
        use_synonyms: bool = True,
    ) -> None:
        self.index = index
        self.index_path = Path(index_path) if index_path else None
        self.retriever = Retriever(index, use_synonyms=use_synonyms)

    # ================================================================== #
    # 构建 / 加载
    # ================================================================== #
    @classmethod
    def build(
        cls,
        corpus_dir: str | Path | None = None,
        index_path: str | Path | None = None,
        *,
        chunk_cfg: ChunkConfig | None = None,
        bm25_cfg: BM25Config | None = None,
        dense: bool = False,
        dense_model: str = DEFAULT_MODEL,
        verbose: bool = False,
    ) -> InfusionPumpRAG:
        corpus_dir = Path(corpus_dir or DEFAULT_CORPUS_DIR)
        index_path = Path(index_path or DEFAULT_INDEX_PATH)
        chunk_cfg = chunk_cfg or ChunkConfig()
        bm25_cfg = bm25_cfg or BM25Config()

        def log(message: str) -> None:
            if verbose:
                print(message)

        log(f"[1/4] 读取语料 {corpus_dir}")
        documents: list[Document] = load_corpus(corpus_dir, chunk_cfg)
        chunks = [c for doc in documents for c in doc.chunks]
        if not chunks:
            raise RuntimeError(f"语料目录里没有可用文档：{corpus_dir}")
        log(f"      {len(documents)} 篇文档 → {len(chunks)} 个片段")

        log("[2/4] 建立 BM25 索引")
        bm25 = BM25Index(k1=bm25_cfg.k1, b=bm25_cfg.b, heading_weight=bm25_cfg.heading_weight).fit(
            [tokenize_for_index(c.text) for c in chunks],
            [tokenize_for_index(c.heading) for c in chunks],
        )
        log(f"      词表 {len(bm25.terms)} 个 term")

        dense_vectors = None
        dense_model_name = ""
        if dense:
            log(f"[3/4] 生成稠密向量（{dense_model}）")
            encoder = DenseEncoder(dense_model)
            dense_vectors = encoder.encode([c.text for c in chunks])
            dense_model_name = encoder.model_name
            log(f"      维度 {encoder.dim}")
        else:
            log("[3/4] 跳过稠密向量（未指定 --dense）")

        index = KnowledgeIndex(
            chunks=chunks,
            docs={doc.meta.doc_id: doc.meta for doc in documents},
            bm25=bm25,
            dense_vectors=dense_vectors,
            dense_model=dense_model_name,
            meta={
                "name": "infusion-pump-rag",
                "version": __version__,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "corpus_dir": corpus_dir.name,
                "n_documents": len(documents),
                "n_chunks": len(chunks),
                "vocab_size": len(bm25.terms),
                "documents": [doc.meta.doc_id for doc in documents],
                "dense_model": dense_model_name,
                "chunk_config": chunk_cfg.to_dict(),
                "bm25_config": bm25_cfg.to_dict(),
            },
        )

        log(f"[4/4] 写入 {index_path}")
        index.save(index_path)
        log("      完成")
        return cls(index, index_path=index_path)

    @classmethod
    def load(
        cls, index_path: str | Path | None = None, *, use_synonyms: bool = True
    ) -> InfusionPumpRAG:
        path = Path(index_path or DEFAULT_INDEX_PATH)
        return cls(KnowledgeIndex.load(path), index_path=path, use_synonyms=use_synonyms)

    # ================================================================== #
    # 查询
    # ================================================================== #
    def search(self, query: str, top_k: int = 4, mode: str = "bm25") -> list[Hit]:
        return self.retriever.search(query, top_k=top_k, mode=mode)

    def ask(
        self,
        query: str,
        top_k: int = 4,
        mode: str = "bm25",
        *,
        use_llm: bool | None = None,
        llm: LLMConfig | None = None,
    ) -> dict:
        """检索并按需生成。use_llm=None 表示"配了大模型就用"。"""
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
        }

    # ================================================================== #
    # 评测
    # ================================================================== #
    def evaluate(
        self,
        questions_path: str | Path | None = None,
        *,
        mode: str = "bm25",
        top_k: int = 5,
    ) -> EvalReport:
        questions = load_questions(questions_path or DEFAULT_EVAL_PATH)
        return evaluate(self, questions, mode=mode, top_k=top_k)

    # ================================================================== #
    # 统计
    # ================================================================== #
    def stats(self) -> dict:
        chunks = self.index.chunks
        data = dict(self.index.meta)
        data.update(
            {
                "index_path": str(self.index_path) if self.index_path else "",
                "index_size_kb": round(self.index_path.stat().st_size / 1024, 1)
                if self.index_path and self.index_path.exists()
                else 0,
                "n_documents": len(self.index.docs),
                "n_chunks": len(chunks),
                "vocab_size": self.index.vocab_size,
                "avg_chunk_chars": round(sum(c.n_chars for c in chunks) / max(len(chunks), 1), 1),
                "min_chunk_chars": min((c.n_chars for c in chunks), default=0),
                "max_chunk_chars": max((c.n_chars for c in chunks), default=0),
                "n_sections": len({(c.doc_id, c.heading) for c in chunks}),
                "has_dense": self.index.has_dense,
                "modes": list(MODES) if self.index.has_dense else ["bm25"],
            }
        )
        return data

    def documents(self) -> list[dict]:
        return [
            {
                "doc_id": meta.doc_id,
                "title": meta.title,
                "path": meta.path,
                "summary": meta.summary,
                "updated": meta.updated,
                "n_chunks": meta.n_chunks,
                "n_chars": meta.n_chars,
                "sources": [s.to_dict() for s in meta.sources],
            }
            for meta in sorted(self.index.docs.values(), key=lambda m: m.path)
        ]
