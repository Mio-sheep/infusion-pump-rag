"""基础测试：python -m unittest discover -s tests -v"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infusion_rag.chunker import split_text
from infusion_rag.config import ChunkConfig, DEFAULT_INDEX_PATH
from infusion_rag.embedder import TfidfVectorizer, dot_sparse
from infusion_rag.pipeline import InfusionPumpRAG
from infusion_rag.tokenizer import tokenize


class TestTokenizer(unittest.TestCase):
    def test_cjk_unigrams_and_bigrams(self):
        tokens = tokenize("输液泵")
        self.assertIn("输", tokens)
        self.assertIn("输液", tokens)
        self.assertIn("液泵", tokens)

    def test_latin_and_units(self):
        tokens = tokenize("流速 4.5 mL/h，符合 IEC 60601-2-24")
        self.assertIn("ml/h", tokens)
        self.assertIn("4.5", tokens)
        self.assertIn("60601-2-24", tokens)

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class TestChunker(unittest.TestCase):
    def test_short_text_single_chunk(self):
        cfg = ChunkConfig(max_chars=100, overlap_chars=10)
        self.assertEqual(len(split_text("很短的一段话。", cfg)), 1)

    def test_long_text_is_split_and_bounded(self):
        cfg = ChunkConfig(max_chars=200, overlap_chars=40)
        text = "".join(f"这是第{i}句测试内容，用于验证分块逻辑是否正确。" for i in range(40))
        chunks = split_text(text, cfg)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), cfg.max_chars + cfg.overlap_chars)

    def test_table_keeps_header(self):
        cfg = ChunkConfig(max_chars=200, overlap_chars=20)
        rows = ["| 项目 | 说明 |", "| --- | --- |"]
        rows += [f"| 项目{i} | 这是第{i}项的很长很长的说明文字内容 |" for i in range(20)]
        chunks = split_text("\n".join(rows), cfg)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("| 项目 | 说明 |", chunk)


class TestVectorizer(unittest.TestCase):
    def test_vectors_normalized_and_ranked(self):
        texts = ["输液泵阻塞报警的处理方法", "注射泵的活塞与推杆结构", "电池维护与更换周期"]
        vec = TfidfVectorizer().fit(texts)
        vectors = vec.encode(texts)
        self.assertEqual(len(vectors), 3)
        for v in vectors:
            norm = sum(w * w for w in v.values()) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=6)

        query = vec.encode(["阻塞报警"])[0]
        scores = [dot_sparse(query, v) for v in vectors]
        self.assertEqual(scores.index(max(scores)), 0)

    def test_state_roundtrip(self):
        vec = TfidfVectorizer().fit(["输液泵", "注射泵"])
        restored = TfidfVectorizer.from_state(vec.state())
        self.assertEqual(restored.terms, vec.terms)
        self.assertEqual(restored.encode(["输液泵"]), vec.encode(["输液泵"]))


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        corpus = root / "raw"
        corpus.mkdir()

        (corpus / "a.md").write_text(
            "# 阻塞报警\n\n## 原因\n\n管路打折或受压会导致阻塞报警。"
            "解除阻塞的瞬间可能出现一次性团注，需要结合临床评估。\n",
            encoding="utf-8",
        )
        (corpus / "b.md").write_text(
            "# 电池维护\n\n## 更换周期\n\n电池应按推荐寿命周期更换，鼓包即为失效信号。\n",
            encoding="utf-8",
        )
        cls.index_path = root / "kb_index.json"
        cls.rag = InfusionPumpRAG.build(
            corpus_dir=corpus, index_path=cls.index_path, verbose=False
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_stats(self):
        stats = self.rag.stats()
        self.assertEqual(stats["n_documents"], 2)
        self.assertGreaterEqual(stats["n_chunks"], 2)
        self.assertFalse(stats["has_dense"])

    def test_search_finds_right_document(self):
        hits = self.rag.search("阻塞报警", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.doc_id, "a")

    def test_synonym_expansion_helps_english(self):
        hits = self.rag.search("电池 battery", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.doc_id, "b")

    def test_ask_extractive(self):
        result = self.rag.ask("阻塞报警的原因", top_k=2, use_llm=False)
        self.assertEqual(result["generator"], "extractive")
        self.assertIn("阻塞", result["answer"])
        self.assertTrue(result["citations"])

    def test_index_reload(self):
        reloaded = InfusionPumpRAG.load(self.index_path)
        self.assertEqual(len(reloaded.index), len(self.rag.index))
        self.assertEqual(
            reloaded.search("阻塞报警", top_k=1)[0].chunk.chunk_id,
            self.rag.search("阻塞报警", top_k=1)[0].chunk.chunk_id,
        )


class TestShippedIndex(unittest.TestCase):
    """如果仓库里已经带了构建好的索引，就顺带验证一下它是可用的。"""

    def test_shipped_index_loads(self):
        if not Path(DEFAULT_INDEX_PATH).exists():
            self.skipTest("仓库中未包含预构建索引")
        rag = InfusionPumpRAG.load(DEFAULT_INDEX_PATH)
        self.assertGreater(len(rag.index), 10)
        hits = rag.search("阻塞报警", top_k=3)
        self.assertTrue(hits)
        self.assertGreater(hits[0].score, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
