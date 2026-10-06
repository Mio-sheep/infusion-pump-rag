"""BM25 检索。

采用字段化（fielded）BM25：正文与小节标题分开建统计量，标题命中额外加权。
这样"标题里带某关键词"就是一个稳定的先验，而不是事后往分数上打补丁。

BM25 相对 TF-IDF 余弦的好处在这里很实际：
  - 词频饱和（k1）抑制了长片段里关键词反复出现造成的虚高；
  - 长度归一化（b）让长短片段可比；
  - IDF 用 N/df 的概率形式，罕见术语（如 "jjf"、"occlusion"）权重更合理。

打分公式（对每个文档 d、每个查询词 t）：

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    tf'(t) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    score(d) = Σ_t idf(t) * tf'(t)  [+ w * 同样的计算，但用标题字段]

注意：这里对每个文档做一次 O(查询词数) 的累加，而不是用倒排表。
在本项目的数据规模（几十到几千个片段）下这完全够用，而且代码直观得多。
"""
from __future__ import annotations

import math
from collections import Counter

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75
DEFAULT_HEADING_WEIGHT = 2.0


class BM25Index:
    """字段化 BM25 索引。"""

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        heading_weight: float = DEFAULT_HEADING_WEIGHT,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.heading_weight = heading_weight

        self.terms: list[str] = []
        self.term_index: dict[str, int] = {}
        self.idf: list[float] = []

        self.body_postings: list[dict[int, int]] = []
        self.heading_postings: list[dict[int, int]] = []
        self.body_len: list[int] = []
        self.heading_len: list[int] = []
        self.avg_body_len = 0.0
        self.avg_heading_len = 0.0
        self.n_docs = 0

    # ------------------------------------------------------------------ #
    # 建索引
    # ------------------------------------------------------------------ #
    def fit(
        self,
        bodies: list[list[str]],
        headings: list[list[str]],
    ) -> "BM25Index":
        """bodies / headings 是已经分好词的 token 序列，两者长度必须一致。"""
        if len(bodies) != len(headings):
            raise ValueError("bodies 与 headings 数量不一致")

        self.n_docs = len(bodies)
        df: Counter[str] = Counter()
        body_postings: list[dict[int, int]] = []
        heading_postings: list[dict[int, int]] = []

        for body, heading in zip(bodies, headings):
            body_counts = Counter(body)
            heading_counts = Counter(heading)
            body_postings.append(body_counts)
            heading_postings.append(heading_counts)
            df.update(body_counts.keys() | heading_counts.keys())

        # 词表按 (文档频率降序, 词形升序) 排列，保证索引构建可复现
        ordered = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))
        self.terms = [term for term, _ in ordered]
        self.term_index = {term: i for i, term in enumerate(self.terms)}

        n = max(self.n_docs, 1)
        self.idf = [
            math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for _, freq in ordered
        ]

        index_of = self.term_index
        self.body_postings = [
            {index_of[t]: c for t, c in counts.items()} for counts in body_postings
        ]
        self.heading_postings = [
            {index_of[t]: c for t, c in counts.items()} for counts in heading_postings
        ]
        self.body_len = [sum(counts.values()) for counts in body_postings]
        self.heading_len = [sum(counts.values()) for counts in heading_postings]
        self.avg_body_len = sum(self.body_len) / n
        self.avg_heading_len = sum(self.heading_len) / n
        return self

    # ------------------------------------------------------------------ #
    # 打分
    # ------------------------------------------------------------------ #
    def score(self, query_tokens: list[str]) -> list[float]:
        """返回每个片段的分数，顺序与建索引时一致。"""
        term_ids = [
            self.term_index[t]
            for t in dict.fromkeys(query_tokens)  # 查询词去重，保持顺序
            if t in self.term_index
        ]
        if not term_ids:
            return [0.0] * self.n_docs

        scores = [0.0] * self.n_docs
        for doc in range(self.n_docs):
            body = self.body_postings[doc]
            heading = self.heading_postings[doc]
            total = 0.0
            for tid in term_ids:
                idf = self.idf[tid]
                body_tf = body.get(tid, 0)
                if body_tf:
                    total += idf * self._saturation(
                        body_tf, self.body_len[doc], self.avg_body_len
                    )
                heading_tf = heading.get(tid, 0)
                if heading_tf:
                    total += (
                        self.heading_weight
                        * idf
                        * self._saturation(
                            heading_tf, self.heading_len[doc], self.avg_heading_len
                        )
                    )
            scores[doc] = total
        return scores

    def _saturation(self, tf: int, length: int, avg_length: float) -> float:
        norm = 1.0 - self.b + self.b * (length / avg_length if avg_length else 1.0)
        return tf * (self.k1 + 1.0) / (tf + self.k1 * norm)

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "heading_weight": self.heading_weight,
            "terms": self.terms,
            "idf": self.idf,
            "body": [{str(k): v for k, v in p.items()} for p in self.body_postings],
            "heading": [{str(k): v for k, v in p.items()} for p in self.heading_postings],
            "body_len": self.body_len,
            "heading_len": self.heading_len,
        }

    @classmethod
    def from_state(cls, state: dict) -> "BM25Index":
        obj = cls(
            k1=float(state.get("k1", DEFAULT_K1)),
            b=float(state.get("b", DEFAULT_B)),
            heading_weight=float(state.get("heading_weight", DEFAULT_HEADING_WEIGHT)),
        )
        obj.terms = list(state["terms"])
        obj.term_index = {term: i for i, term in enumerate(obj.terms)}
        obj.idf = [float(x) for x in state["idf"]]
        obj.body_postings = [
            {int(k): int(v) for k, v in p.items()} for p in state["body"]
        ]
        obj.heading_postings = [
            {int(k): int(v) for k, v in p.items()} for p in state["heading"]
        ]
        obj.body_len = [int(x) for x in state["body_len"]]
        obj.heading_len = [int(x) for x in state["heading_len"]]
        n = max(len(obj.body_len), 1)
        obj.avg_body_len = sum(obj.body_len) / n
        obj.avg_heading_len = sum(obj.heading_len) / n
        obj.n_docs = len(obj.body_len)
        return obj

    def __len__(self) -> int:
        return self.n_docs
