"""前置元数据解析、分块边界、表格处理、出处收集。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infusion_rag.chunker import load_corpus, parse_document, split_section
from infusion_rag.config import ChunkConfig
from infusion_rag.frontmatter import split_front_matter
from tests.fixtures import BATTERY_DOC, OCCLUSION_DOC, temp_project


class TestFrontMatter(unittest.TestCase):
    def test_parses_scalars_and_object_list(self):
        meta, body = split_front_matter(OCCLUSION_DOC)
        self.assertEqual(meta["title"], "阻塞报警处置")
        self.assertEqual(meta["updated"], "2026-02-28")
        self.assertEqual(len(meta["sources"]), 2)
        self.assertEqual(
            meta["sources"][0]["label"], "FDA, Examples of Reported Infusion Pump Problems"
        )
        self.assertNotIn("---", body.splitlines()[0])
        self.assertTrue(body.lstrip().startswith("# 阻塞报警处置"))

    def test_plain_markdown_passes_through(self):
        meta, body = split_front_matter("# 标题\n\n正文")
        self.assertEqual(meta, {})
        self.assertEqual(body, "# 标题\n\n正文")

    def test_unterminated_fence_is_treated_as_body(self):
        text = "---\ntitle: x\n\n# 标题"
        meta, body = split_front_matter(text)
        self.assertEqual(meta, {})
        self.assertEqual(body, text)


class TestChunking(unittest.TestCase):
    def test_short_section_stays_whole(self):
        cfg = ChunkConfig(max_chars=500, overlap_chars=50)
        self.assertEqual(len(split_section("很短的一段话。", cfg)), 1)

    def test_long_section_is_split_within_bounds(self):
        cfg = ChunkConfig(max_chars=200, overlap_chars=40, min_chars=1)
        text = "".join(f"这是第{i}句测试内容，用来验证分块逻辑。" for i in range(60))
        pieces = split_section(text, cfg)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertLessEqual(len(piece), cfg.max_chars + cfg.overlap_chars)

    def test_table_blocks_repeat_the_header(self):
        cfg = ChunkConfig(max_chars=220, overlap_chars=20, min_chars=1)
        rows = ["| 档位 | 说明 |", "| --- | --- |"]
        rows += [f"| 档位{i} | 这是第 {i} 档的说明文字 |" for i in range(24)]
        pieces = split_section("\n".join(rows), cfg)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertIn("| 档位 | 说明 |", piece)
            self.assertIn("| --- | --- |", piece)


class TestDocumentParsing(unittest.TestCase):
    def test_metadata_and_heading_path(self):
        with temp_project() as (corpus, _):
            doc = parse_document(corpus / "01-阻塞报警处置.md", corpus, ChunkConfig())
        self.assertEqual(doc.meta.doc_id, "01-阻塞报警处置")
        self.assertEqual(doc.meta.title, "阻塞报警处置")
        self.assertEqual(len(doc.meta.sources), 2)
        headings = {c.heading for c in doc.chunks}
        self.assertIn("上游阻塞", headings)
        self.assertIn("压力档位对照表", headings)

    def test_source_section_is_not_indexed(self):
        with temp_project() as (corpus, _):
            doc = parse_document(corpus / "01-阻塞报警处置.md", corpus, ChunkConfig())
        for chunk in doc.chunks:
            self.assertNotIn("参考来源", chunk.heading)
            self.assertNotIn("example.org", chunk.text)

    def test_sources_are_inherited_from_body_when_no_frontmatter(self):
        """没有前置元数据时，仍然能从正文的"参考来源"小节里捡到出处。"""
        with temp_project() as (corpus, _):
            path = corpus / "03-无前置元数据.md"
            path.write_text(
                "# 无前置元数据\n\n## 正文\n\n这里是一些内容。\n\n"
                "## 参考来源\n\n- [某标准](https://example.org/std)\n",
                encoding="utf-8",
            )
            doc = parse_document(path, corpus, ChunkConfig())
        self.assertEqual([s.url for s in doc.meta.sources], ["https://example.org/std"])
        self.assertNotIn("参考来源", {c.heading for c in doc.chunks})

    def test_title_falls_back_to_h1_then_filename(self):
        with temp_project() as (corpus, _):
            path = corpus / "04-无标题.md"
            path.write_text("# 来自 H1 的标题\n\n正文内容。\n", encoding="utf-8")
            doc = parse_document(path, corpus, ChunkConfig())
        self.assertEqual(doc.meta.title, "来自 H1 的标题")

    def test_chunk_ids_are_unique_and_ordered_within_a_document(self):
        with temp_project() as (corpus, _):
            docs = load_corpus(corpus, ChunkConfig())
        ids = [c.chunk_id for doc in docs for c in doc.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        for doc in docs:
            orders = [c.order for c in doc.chunks]
            self.assertEqual(
                orders,
                list(range(1, len(orders) + 1)),
                f"{doc.meta.doc_id} 的 order 不是从 1 连续递增",
            )

    def test_corpus_is_read_in_filename_order(self):
        with temp_project() as (corpus, _):
            docs = load_corpus(corpus, ChunkConfig())
        self.assertEqual([d.meta.path for d in docs], ["01-阻塞报警处置.md", "02-电池与电源.md"])

    def test_missing_corpus_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_corpus(Path("definitely/not/here"))


class TestBatchDocCounts(unittest.TestCase):
    def test_meta_counts_match_chunks(self):
        with temp_project() as (corpus, _):
            doc = parse_document(corpus / "02-电池与电源.md", corpus, ChunkConfig())
        self.assertEqual(doc.meta.n_chunks, len(doc.chunks))
        self.assertEqual(doc.meta.n_chars, sum(c.n_chars for c in doc.chunks))
        self.assertLess(len(BATTERY_DOC), 2000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
