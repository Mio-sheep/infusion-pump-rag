"""端到端：构建索引 → 检索 → 问答 → 统计。"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infusion_rag import InfusionPumpRAG
from infusion_rag.config import BM25Config, ChunkConfig
from infusion_rag.store import KnowledgeIndex, dense_path_for
from tests.fixtures import temp_project, temp_rag


class TestBuild(unittest.TestCase):
    def test_build_writes_index_and_loads_back(self):
        with temp_rag() as (rag, _, index_path):
            self.assertTrue(index_path.exists())
            self.assertFalse(dense_path_for(index_path).exists())
            reloaded = InfusionPumpRAG.load(index_path)
        self.assertEqual(len(reloaded.index), len(rag.index))
        self.assertEqual(
            [h.chunk.chunk_id for h in reloaded.search("阻塞", top_k=3)],
            [h.chunk.chunk_id for h in rag.search("阻塞", top_k=3)],
        )

    def test_index_is_valid_json_with_expected_sections(self):
        with temp_rag() as (_, _, index_path):
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], 3)
        for key in ("meta", "docs", "chunks", "bm25"):
            self.assertIn(key, payload)
        self.assertEqual(len(payload["docs"]), 2)

    def test_empty_corpus_raises(self):
        with temp_project() as (corpus, index_path):
            for path in corpus.iterdir():
                path.unlink()
            with self.assertRaises(RuntimeError):
                InfusionPumpRAG.build(corpus_dir=corpus, index_path=index_path)

    def test_bm25_config_reaches_the_index(self):
        with temp_rag(bm25_cfg=BM25Config(heading_weight=5.0)) as (rag, _, _):
            self.assertEqual(rag.index.bm25.heading_weight, 5.0)

    def test_chunk_config_reaches_the_index(self):
        cfg = ChunkConfig(max_chars=120, overlap_chars=30, min_chars=10)
        with temp_rag(chunk_cfg=cfg) as (rag, _, _):
            self.assertLessEqual(rag.stats()["max_chunk_chars"], 200)


class TestQuery(unittest.TestCase):
    def test_ask_extractive_mentions_sources(self):
        with temp_rag() as (rag, _, _):
            result = rag.ask("下游阻塞", top_k=2, use_llm=False)
        self.assertEqual(result["generator"], "extractive")
        self.assertIn("下游阻塞", result["answer"])
        self.assertIn("example.org", result["answer"])
        self.assertEqual(result["n_hits"], len(result["hits"]))

    def test_ask_with_no_hits_is_explicit(self):
        with temp_rag() as (rag, _, _):
            result = rag.ask("zzzq wwwx", top_k=3, use_llm=False)
        self.assertEqual(result["n_hits"], 0)
        self.assertIn("没有检索到", result["answer"])

    def test_ask_json_is_serialisable(self):
        with temp_rag() as (rag, _, _):
            result = rag.ask("电池", top_k=2, use_llm=False)
        json.dumps(result, ensure_ascii=False)


class TestMetadataViews(unittest.TestCase):
    def test_stats_have_the_fields_the_cli_prints(self):
        with temp_rag() as (rag, _, _):
            stats = rag.stats()
        for key in (
            "n_documents", "n_chunks", "n_sections", "vocab_size",
            "avg_chunk_chars", "max_chunk_chars", "has_dense", "modes",
        ):
            self.assertIn(key, stats)
        self.assertEqual(stats["n_documents"], 2)
        self.assertEqual(stats["modes"], ["bm25"])

    def test_documents_view_exposes_sources(self):
        with temp_rag() as (rag, _, _):
            docs = rag.documents()
        self.assertEqual([d["path"] for d in docs],
                         ["01-阻塞报警处置.md", "02-电池与电源.md"])
        self.assertTrue(all(d["sources"] for d in docs))
        self.assertTrue(all(d["n_chunks"] > 0 for d in docs))

    def test_missing_index_gives_actionable_error(self):
        with temp_project() as (_, index_path):
            with self.assertRaises(FileNotFoundError) as ctx:
                KnowledgeIndex.load(index_path)
        self.assertIn("build", str(ctx.exception))

    def test_index_version_mismatch_is_reported(self):
        with temp_rag() as (_, _, index_path):
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            payload["version"] = 1
            index_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                KnowledgeIndex.load(index_path)
        self.assertIn("重新运行", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
