"""分词与索引前文本规整。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infusion_rag.tokenizer import normalize, tokenize, tokenize_for_index, unique_tokens


class TestNormalize(unittest.TestCase):
    def test_strips_urls_but_keeps_link_text(self):
        text = "参见 [FDA 的报告](https://www.fda.gov/medical-devices/infusion-pumps)。"
        cleaned = normalize(text)
        self.assertIn("FDA 的报告", cleaned)
        self.assertNotIn("http", cleaned)

    def test_keeps_inline_code_content(self):
        self.assertEqual(normalize("流速设为 `4.5 mL/h`").strip(), "流速设为 4.5 mL/h")

    def test_removes_emphasis_markers(self):
        cleaned = normalize("**重点**和 *次要*")
        self.assertNotIn("*", cleaned)
        self.assertIn("重点", cleaned)
        self.assertIn("次要", cleaned)


class TestTokenize(unittest.TestCase):
    def test_cjk_yields_unigrams_and_bigrams(self):
        tokens = tokenize("阻塞报警")
        for expected in ("阻", "阻塞", "塞报", "报警", "警"):
            self.assertIn(expected, tokens)

    def test_latin_keeps_units_and_standard_numbers(self):
        tokens = tokenize("4.5 mL/h 符合 IEC 60601-2-24")
        self.assertIn("4.5", tokens)
        self.assertIn("ml/h", tokens)
        self.assertIn("60601-2-24", tokens)

    def test_lowercases(self):
        self.assertIn("occlusion", tokenize("Occlusion"))

    def test_empty_input(self):
        self.assertEqual(tokenize(""), [])

    def test_tokenize_for_index_drops_url_tokens(self):
        tokens = tokenize_for_index("[JJF 1259](https://www.ndls.org.cn/standard/detail/abc)")
        self.assertIn("jjf", tokens)
        self.assertIn("1259", tokens)
        self.assertNotIn("ndls", tokens)

    def test_unique_tokens_preserves_order(self):
        # "阻塞" 先产出单字再产出二元组，所以顺序是 阻、塞、阻塞
        tokens = unique_tokens("阻塞 阻塞 阻塞")
        self.assertEqual(tokens, ["阻", "塞", "阻塞"])
        self.assertEqual(len(tokens), len(set(tokens)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
