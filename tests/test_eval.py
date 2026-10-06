"""评测指标本身的正确性。指标写错的评测比没有评测更糟。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infusion_rag.chunker import Chunk
from infusion_rag.eval import (
    EvalReport,
    Expectation,
    Question,
    QuestionResult,
    evaluate,
    load_questions,
)


def make_chunk(doc_id: str, heading: str = "") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#001",
        doc_id=doc_id,
        doc_title=doc_id,
        path=f"{doc_id}.md",
        heading=heading,
        text="正文",
        order=1,
        n_chars=2,
    )


class StubHit:
    def __init__(self, chunk, rank):
        self.chunk = chunk
        self.rank = rank
        self.label = chunk.heading or chunk.doc_id


class StubRag:
    """按预设顺序返回命中的假检索器。"""

    def __init__(self, plan: dict[str, list[Chunk]]):
        self.plan = plan

    def search(self, query, top_k=5, mode="bm25"):
        return [StubHit(c, i) for i, c in enumerate(self.plan.get(query, []), 1)][:top_k]


class TestExpectation(unittest.TestCase):
    def test_doc_only(self):
        self.assertTrue(Expectation(doc="a").matches(make_chunk("a")))
        self.assertFalse(Expectation(doc="a").matches(make_chunk("b")))

    def test_section_substring(self):
        expectation = Expectation(doc="a", section="阻塞")
        self.assertTrue(expectation.matches(make_chunk("a", "2.1 阻塞报警")))
        self.assertFalse(expectation.matches(make_chunk("a", "2.2 气泡报警")))

    def test_matches_on_title_too(self):
        chunk = make_chunk("a")
        chunk.doc_title = "阻塞报警处置"
        self.assertTrue(Expectation(doc="阻塞报警处置").matches(chunk))


class TestMetrics(unittest.TestCase):
    def _report(self, ranks_per_question):
        results = []
        for i, ranks in enumerate(ranks_per_question):
            question = Question(qid=f"q{i}", question=f"问题{i}", expect=[])
            results.append(QuestionResult(question=question, ranks=list(ranks)))
        return EvalReport(mode="bm25", top_k=5, results=results)

    def test_recall_at_k(self):
        report = self._report([[1], [3], [4], []])
        self.assertEqual(report.recall_at(1), 0.25)
        self.assertEqual(report.recall_at(3), 0.5)
        self.assertEqual(report.recall_at(5), 0.75)

    def test_mrr(self):
        report = self._report([[1], [2], [4], []])
        expected = (1 / 1 + 1 / 2 + 1 / 4 + 0) / 4
        self.assertAlmostEqual(report.mrr(), expected, places=6)

    def test_first_rank_takes_the_best(self):
        self.assertEqual(self._report([[4, 2]]).results[0].first_rank, 2)

    def test_empty_report_is_zero_not_crash(self):
        report = self._report([])
        self.assertEqual(report.mrr(), 0.0)
        self.assertEqual(report.recall_at(1), 0.0)

    def test_summary_keys(self):
        summary = self._report([[1]]).summary()
        self.assertEqual(summary["questions"], 1)
        self.assertIn("recall@1", summary)
        self.assertIn("recall@5", summary)
        self.assertIn("mrr", summary)

    def test_markdown_renders(self):
        text = self._report([[1], []]).to_markdown(show_failures=5)
        self.assertIn("Recall@1", text)
        self.assertIn("MRR", text)
        self.assertIn("q1", text)


class TestEvaluateLoop(unittest.TestCase):
    def test_hits_and_misses(self):
        rag = StubRag(
            {
                "阻塞": [make_chunk("01", "2.1 阻塞报警")],
                "气泡": [make_chunk("01", "2.2 气泡报警")],
            }
        )
        questions = [
            Question(qid="a", question="阻塞", expect=[Expectation("01", "阻塞")]),
            Question(qid="b", question="气泡", expect=[Expectation("02", "气泡")]),
        ]
        report = evaluate(rag, questions, mode="bm25", top_k=5)
        self.assertEqual(report.n, 2)
        self.assertEqual(report.recall_at(1), 0.5)
        self.assertEqual([r.question.qid for r in report.failures(1)], ["b"])


class TestLoadQuestions(unittest.TestCase):
    def test_parses_jsonl_with_comments_and_blanks(self):
        payload = (
            "// 这是注释\n"
            "\n"
            '{"id":"q1","question":"阻塞报警怎么排查",'
            '"expect":[{"doc":"01","section":"阻塞"}],"note":"基础"}'
            "\n"
            '{"id":"q2","question":"简单问题","expect":["02"]}\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.jsonl"
            path.write_text(payload, encoding="utf-8")
            questions = load_questions(path)

        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0].expect[0].doc, "01")
        self.assertEqual(questions[0].expect[0].section, "阻塞")
        self.assertEqual(questions[1].expect[0].doc, "02")
        self.assertEqual(questions[1].expect[0].section, "")

    def test_bad_line_names_the_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.jsonl"
            path.write_text('{"id":"q1"}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_questions(path)
        self.assertIn("q.jsonl:1", str(ctx.exception))

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_questions("no/such/questions.jsonl")

    def test_shipped_eval_set_is_wellformed(self):
        from infusion_rag.config import DEFAULT_EVAL_PATH

        if not Path(DEFAULT_EVAL_PATH).exists():
            self.skipTest("仓库中没有评测集")
        questions = load_questions(DEFAULT_EVAL_PATH)
        self.assertGreaterEqual(len(questions), 30)
        ids = [q.qid for q in questions]
        self.assertEqual(len(ids), len(set(ids)), "评测集里的 id 有重复")
        for question in questions:
            self.assertTrue(question.expect, f"{question.qid} 没有标注期望")


if __name__ == "__main__":
    unittest.main(verbosity=2)
