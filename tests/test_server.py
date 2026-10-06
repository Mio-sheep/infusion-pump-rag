"""网页版接口。用 port=0 起随机端口，避免和别的进程撞车。"""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infusion_rag.server import MODE_LABELS, make_server, render_page
from tests.fixtures import temp_rag


class ServerTestCase(unittest.TestCase):
    """起一个真服务，跑完关掉。"""

    @classmethod
    def setUpClass(cls):
        cls._ctx = temp_rag()
        cls.rag, _, _ = cls._ctx.__enter__()
        cls.httpd = make_server(cls.rag, "127.0.0.1", 0, quiet=True)
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        cls._ctx.__exit__(None, None, None)

    def get(self, path: str):
        with urllib.request.urlopen(f"{self.base}{path}", timeout=10) as response:
            return response.status, response.read().decode("utf-8")

    def get_json(self, path: str):
        status, body = self.get(path)
        return status, json.loads(body)


class TestPage(ServerTestCase):
    def test_root_serves_html_with_mode_options(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("输液泵 / 注射泵 知识库", body)
        self.assertIn('value="bm25"', body)
        self.assertNotIn("__MODE_OPTIONS__", body, "占位符没有被替换掉")

    def test_render_page_only_lists_available_modes(self):
        html = render_page(["bm25"])
        self.assertNotIn('value="dense"', html)
        html_all = render_page(["bm25", "dense", "hybrid"])
        for mode in ("bm25", "dense", "hybrid"):
            self.assertIn(f'value="{mode}"', html_all)
        self.assertEqual(set(MODE_LABELS), {"bm25", "dense", "hybrid"})

    def test_unknown_path_is_404_json(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/api/nope")
        self.assertEqual(ctx.exception.code, 404)


class TestApi(ServerTestCase):
    def test_stats(self):
        status, data = self.get_json("/api/stats")
        self.assertEqual(status, 200)
        self.assertEqual(data["n_documents"], 2)
        self.assertIn("llm_enabled", data)

    def test_docs(self):
        _, data = self.get_json("/api/docs")
        self.assertEqual(len(data["documents"]), 2)
        self.assertTrue(data["documents"][0]["sources"])

    def test_search(self):
        _, data = self.get_json("/api/search?q=%E9%98%BB%E5%A1%9E&k=2")
        self.assertEqual(data["mode"], "bm25")
        self.assertGreater(data["n_hits"], 0)
        hit = data["hits"][0]
        for key in ("rank", "doc_title", "heading", "path", "score", "text", "sources"):
            self.assertIn(key, hit)

    def test_ask_uses_extractive_without_llm(self):
        _, data = self.get_json("/api/ask?q=%E7%94%B5%E6%B1%A0&k=2&llm=0")
        self.assertEqual(data["generator"], "extractive")
        self.assertIn("电池", data["answer"])

    def test_missing_query_is_400(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/api/search")
        self.assertEqual(ctx.exception.code, 400)
        error = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertIn("q", error["error"])

    def test_bad_mode_is_400_with_the_valid_list(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/api/search?q=test&mode=magic")
        error = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertIn("bm25", error["error"])

    def test_top_k_is_clamped(self):
        _, data = self.get_json("/api/search?q=%E9%98%BB%E5%A1%9E&k=99999")
        self.assertLessEqual(len(data["hits"]), 20)

    def test_non_numeric_k_falls_back_to_default(self):
        _, data = self.get_json("/api/search?q=%E9%98%BB%E5%A1%9E&k=abc")
        self.assertLessEqual(len(data["hits"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
