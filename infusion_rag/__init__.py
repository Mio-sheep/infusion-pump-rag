"""输液泵 / 注射泵 小型 RAG 知识库。

核心链路（分块、BM25 检索、抽取式问答、网页界面）只用 Python 标准库实现，
不装任何第三方包就能跑；稠密向量与大模型生成都是可选项。

典型用法：

    from infusion_rag import InfusionPumpRAG

    rag = InfusionPumpRAG.load()                 # 读取 index/kb_index.json
    for hit in rag.search("阻塞报警的原因", top_k=3):
        print(hit.rank, hit.label, hit.score)
"""

from .bm25 import BM25Index
from .chunker import Chunk, Document, load_corpus, parse_document, split_section
from .config import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_EVAL_PATH,
    DEFAULT_INDEX_PATH,
    BM25Config,
    ChunkConfig,
    LLMConfig,
)
from .eval import EvalReport, Question, evaluate, load_questions
from .frontmatter import DocMeta, Source
from .pipeline import InfusionPumpRAG, __version__
from .retriever import Hit, Retriever
from .store import KnowledgeIndex

__all__ = [
    "BM25Index",
    "BM25Config",
    "Chunk",
    "ChunkConfig",
    "DEFAULT_CORPUS_DIR",
    "DEFAULT_EVAL_PATH",
    "DEFAULT_INDEX_PATH",
    "DocMeta",
    "Document",
    "EvalReport",
    "Hit",
    "InfusionPumpRAG",
    "KnowledgeIndex",
    "LLMConfig",
    "Question",
    "Retriever",
    "Source",
    "evaluate",
    "load_corpus",
    "load_questions",
    "parse_document",
    "split_section",
    "__version__",
]
