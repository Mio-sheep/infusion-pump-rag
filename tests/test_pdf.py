"""PDF 语料接入的测试。

pypdf 是可选依赖，没装就整文件跳过（CI 的核心链路刻意不装任何依赖）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import pypdf  # noqa: F401

    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from infusion_rag.chunker import ChunkConfig, parse_document
from infusion_rag.pdf import PdfError, pdf_to_markdown

from .pdf_fixture import build_pdf

SAMPLE_PAGES = [
    [
        ("Infusion Pump Service Manual", 18.0, 720),
        ("The occlusion alarm indicates a blockage in the fluid path.", 11.0, 690),
        ("Check the roller clamp before replacing the tube set.", 11.0, 675),
        ("Chapter 2 Maintenance", 14.0, 640),
        ("Clean the pump housing with a damp cloth weekly.", 11.0, 610),
    ]
]


@unittest.skipUnless(HAS_PYPDF, "需要可选依赖 pypdf")
class TestPdfToMarkdown(unittest.TestCase):
    def _write(self, pages, title=""):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(build_pdf(pages, title=title))
            return Path(fh.name)

    def test_heading_from_font_size(self):
        path = self._write(SAMPLE_PAGES)
        md = pdf_to_markdown(path)
        # 最大字号 → #，次大 → ##
        self.assertIn("# Infusion Pump Service Manual", md)
        self.assertIn("## Chapter 2 Maintenance", md)
        # 正文两行被拼成一段，拉丁词之间补空格
        self.assertIn("path. Check", md)

    def test_pdf_metadata_title_wins(self):
        path = self._write(SAMPLE_PAGES, title="Metadata Title")
        md = pdf_to_markdown(path)
        self.assertIn("title: Metadata Title", md)

    def test_scanned_pdf_raises_helpful_error(self):
        path = self._write([[]])  # 没有任何文字层
        with self.assertRaises(PdfError) as ctx:
            pdf_to_markdown(path)
        self.assertIn("OCR", str(ctx.exception))

    def test_empty_fragment_pages_ignored(self):
        # 有文字的页 + 空页混排：空页不该让整体失败
        path = self._write([SAMPLE_PAGES[0], []])
        md = pdf_to_markdown(path)
        self.assertIn("occlusion alarm", md)


@unittest.skipUnless(HAS_PYPDF, "需要可选依赖 pypdf")
class TestPdfChunking(unittest.TestCase):
    def test_pdf_document_chunks_like_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "12-sample-manual.pdf"
            pdf_path.write_bytes(build_pdf(SAMPLE_PAGES, title="Sample Manual"))

            doc = parse_document(pdf_path, root, ChunkConfig())
            self.assertEqual(doc.meta.title, "Sample Manual")
            headings = {c.heading for c in doc.chunks}
            # 字号推断出的章节进了片段元数据，参与字段化 BM25 打分
            self.assertIn("Chapter 2 Maintenance", headings)
            joined = "\n".join(c.text for c in doc.chunks)
            self.assertIn("occlusion alarm", joined)

    def test_body_split_respects_max_chars(self):
        long_body = " ".join(f"Sentence number {i} about pump safety." for i in range(60))
        pages = [[("Big Title", 18.0, 720), (long_body, 11.0, 690)]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "long.pdf"
            pdf_path.write_bytes(build_pdf(pages))
            doc = parse_document(pdf_path, root, ChunkConfig(max_chars=400))
            self.assertGreater(len(doc.chunks), 1)
            # max_chars 是目标上限，片段间的重叠会让实际长度略超
            cfg = ChunkConfig(max_chars=400)
            for chunk in doc.chunks:
                self.assertLessEqual(chunk.n_chars, cfg.max_chars + cfg.overlap_chars + 40)


if __name__ == "__main__":
    unittest.main()
