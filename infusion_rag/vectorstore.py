"""Chroma 向量库后端（可选）。

为什么要有这个模块：

    现在的稠密向量存在 kb_dense_index.json 里，检索时把问题向量和
    所有片段向量逐个算余弦——暴力扫描。81 个片段毫无压力，
    但语料涨到十万片段时，每次查询都是上亿次乘法。

    Chroma 内部用 HNSW（近似最近邻）把查找从 O(N) 降到近似 O(logN)：
    预先把向量组织成一张图，查询时从入口一路向最近的邻居跳过去。

代价与取舍：

    chromadb 是一个不小的第三方依赖，而这个项目的卖点是"核心零依赖"。
    所以它做成可选后端：默认仍是 JSON + 暴力扫描，--vector-store chroma
    才启用。语料上量那天，换一个参数就能切过来，不用改代码。

数据放哪：

    index/chroma/ 目录（gitignore，可随时从语料重建）。
    每个向量带 metadata：chunk_id / doc_id / 标题 / 小节，
    查询回来直接知道命中了哪个片段。
"""

from __future__ import annotations

from pathlib import Path

from .chunker import Chunk

INSTALL_HINT = (
    "未安装 chromadb。这是可选依赖，安装方式：\n"
    "    pip install chromadb\n"
    "或者继续用默认的 JSON 向量存储（不加 --vector-store chroma）。"
)


class ChromaStore:
    """Chroma 的薄封装：建集合、写入、按向量查最近邻。"""

    def __init__(self, persist_dir: str | Path, collection: str = "infusion_kb") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise RuntimeError(INSTALL_HINT) from exc

        self.persist_dir = str(Path(persist_dir))
        # 匿名客户端 = 纯本地嵌入式模式，不需要起服务
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        # cosine 距离：和项目里手写的余弦相似度同一套度量
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def __len__(self) -> int:
        return self._collection.count()

    def reset(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """全量重建：先清空再写入。构建索引用，检索时不碰这个方法。"""
        if self._collection.count():
            self._client.delete_collection(self._collection.name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection.name, metadata={"hnsw:space": "cosine"}
            )

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[
                {"doc_id": c.doc_id, "doc_title": c.doc_title, "heading": c.heading} for c in chunks
            ],
        )

    def query(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """查最近邻。返回 [(chunk_id, 相似度)]，按相似度从高到低。

        Chroma 返回的是"距离"，cosine 空间下 相似度 = 1 - 距离。
        """
        result = self._collection.query(
            query_embeddings=[list(query_vector)],
            n_results=min(top_k, max(len(self), 1)) or top_k,
        )
        ids = result["ids"][0] if result.get("ids") else []
        distances = result["distances"][0] if result.get("distances") else []
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, distances)]
