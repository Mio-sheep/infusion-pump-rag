"""BM25 与检索融合。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infusion_rag.bm25 import BM25Index
from infusion_rag.tokenizer import tokenize_for_index
from tests.fixtures import temp_rag


def build_index(bodies, headings=None):
    headings = headings or [""] * len(bodies)
    return BM25Index().fit(
        [tokenize_for_index(b) for b in bodies],
        [tokenize_for_index(h) for h in headings],
    )


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.bodies = [
            "管路打折会导致下游阻塞报警，解除阻塞时可能出现团注。",
            "电池应按推荐寿命周期更换，鼓包的电池必须停用。",
            "校准流量误差时使用分析天平称量法，注意记录测量不确定度。",
        ]
        self.index = build_index(self.bodies)

    def test_ranks_the_relevant_document_first(self):
        scores = self.index.score(tokenize_for_index("阻塞报警"))
        self.assertEqual(scores.index(max(scores)), 0)

    def test_query_with_no_known_terms_scores_zero(self):
        scores = self.index.score(tokenize_for_index("zzzq wwwx"))
        self.assertEqual(scores, [0.0, 0.0, 0.0])

    def test_topic_match_beats_incidental_character_overlap(self):
        """中文单字会带来偶然重合，但相关性高的查询必须仍然得分更高。"""
        on_topic = max(self.index.score(tokenize_for_index("阻塞报警")))
        off_topic = max(self.index.score(tokenize_for_index("量子色动力学重整化")))
        self.assertGreater(on_topic, off_topic)

    def test_empty_query_returns_zeros(self):
        self.assertEqual(self.index.score([]), [0.0, 0.0, 0.0])

    def test_heading_field_breaks_ties(self):
        """正文相同、标题不同时，标题命中的那个应该排前面。"""
        bodies = ["这是一段通用说明文字，内容一致。", "这是一段通用说明文字，内容一致。"]
        index = build_index(bodies, headings=["阻塞报警的原因", "电池维护"])
        scores = index.score(tokenize_for_index("阻塞"))
        self.assertGreater(scores[0], scores[1])

    def test_heading_weight_zero_ignores_headings(self):
        bodies = ["通用说明文字。", "通用说明文字。"]
        index = BM25Index(heading_weight=0.0).fit(
            [tokenize_for_index(b) for b in bodies],
            [tokenize_for_index(h) for h in ["阻塞报警", "电池"]],
        )
        self.assertEqual(index.score(tokenize_for_index("阻塞")), [0.0, 0.0])

    def test_longer_document_is_penalized(self):
        """同样出现一次关键词，长文档的分数应低于短文档。"""
        short = "阻塞报警。"
        long_ = "阻塞报警。" + "无关内容。" * 60
        index = build_index([short, long_])
        scores = index.score(tokenize_for_index("阻塞报警"))
        self.assertGreater(scores[0], scores[1])

    def test_state_round_trip_is_exact(self):
        restored = BM25Index.from_state(self.index.state())
        query = tokenize_for_index("电池 鼓包")
        self.assertEqual(restored.score(query), self.index.score(query))
        self.assertEqual(restored.terms, self.index.terms)
        self.assertEqual(restored.n_docs, self.index.n_docs)

    def test_fit_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            BM25Index().fit([["a"]], [["b"], ["c"]])


class TestRetriever(unittest.TestCase):
    def test_bm25_mode_finds_the_right_document(self):
        with temp_rag() as (rag, _, _):
            hits = rag.search("上游阻塞", top_k=3, mode="bm25")
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.doc_id, "01-阻塞报警处置")
        self.assertIn("上游阻塞", hits[0].chunk.heading)

    def test_english_query_matches_chinese_corpus(self):
        """查询扩展要能把英文说法映射到中文正文上。"""
        with temp_rag() as (rag, _, _):
            hits = rag.search("occlusion", top_k=3, mode="bm25")
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.doc_id, "01-阻塞报警处置")

    def test_hits_carry_document_sources(self):
        with temp_rag() as (rag, _, _):
            hits = rag.search("电池", top_k=2)
        urls = {s["url"] for hit in hits for s in hit.sources}
        self.assertIn("https://example.org/fda-clinicians", urls)

    def test_scores_are_sorted_descending(self):
        with temp_rag() as (rag, _, _):
            hits = rag.search("阻塞 电池 校准", top_k=4)
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query_returns_nothing(self):
        with temp_rag() as (rag, _, _):
            self.assertEqual(rag.search("   "), [])

    def test_unknown_mode_is_rejected(self):
        with temp_rag() as (rag, _, _):
            with self.assertRaises(ValueError):
                rag.search("阻塞", mode="magic")

    def test_dense_mode_without_vectors_explains_how_to_fix(self):
        with temp_rag() as (rag, _, _):
            with self.assertRaises(RuntimeError) as ctx:
                rag.search("阻塞", mode="dense")
        self.assertIn("--dense", str(ctx.exception))

    def test_synonyms_can_be_disabled(self):
        """语料正文是中文，关掉扩展后纯英文查询应该什么都检索不到。"""
        with temp_rag() as (rag, _, _):
            self.assertTrue(rag.search("occlusion", top_k=3))
            rag.retriever.use_synonyms = False
            self.assertEqual(rag.search("occlusion", top_k=3), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
