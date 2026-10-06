"""检索效果评测。

一个 RAG 项目如果没有可复现的评测，就没法回答"改了检索之后是变好还是变差"。
这里用一份人工标注的问题集（eval/questions.jsonl）来算召回率与 MRR。

判定标准刻意做成"文档 + 小节"两级，而不是绑定 chunk_id：
重新分块、调整参数之后 chunk_id 会全部变掉，而小节标题是稳定的。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_K_VALUES = (1, 3, 5)


@dataclass
class Expectation:
    """一条"什么才算答对"的判定。section 为空表示只要命中该文档就算对。"""

    doc: str
    section: str = ""

    def matches(self, chunk) -> bool:
        if chunk.doc_id != self.doc and chunk.doc_title != self.doc:
            return False
        if not self.section:
            return True
        return self.section in chunk.heading

    def __str__(self) -> str:
        return f"{self.doc} › {self.section}" if self.section else self.doc


@dataclass
class Question:
    qid: str
    question: str
    expect: list[Expectation]
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        raw = data.get("expect") or []
        expect: list[Expectation] = []
        for item in raw:
            if isinstance(item, str):
                expect.append(Expectation(doc=item))
            else:
                expect.append(
                    Expectation(doc=item.get("doc", ""), section=item.get("section", ""))
                )
        return cls(
            qid=str(data.get("id", "")),
            question=str(data["question"]),
            expect=expect,
            note=str(data.get("note", "")),
        )


@dataclass
class QuestionResult:
    question: Question
    ranks: list[int] = field(default_factory=list)  # 命中期望的片段名次
    top: list[str] = field(default_factory=list)    # 实际 top-k 的定位串

    @property
    def first_rank(self) -> int | None:
        return min(self.ranks) if self.ranks else None

    def hit_at(self, k: int) -> bool:
        return self.first_rank is not None and self.first_rank <= k


@dataclass
class EvalReport:
    mode: str
    top_k: int
    results: list[QuestionResult]
    k_values: tuple[int, ...] = DEFAULT_K_VALUES

    @property
    def n(self) -> int:
        return len(self.results)

    def recall_at(self, k: int) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.hit_at(k)) / len(self.results)

    def mrr(self) -> float:
        if not self.results:
            return 0.0
        total = 0.0
        for result in self.results:
            rank = result.first_rank
            total += 1.0 / rank if rank else 0.0
        return total / len(self.results)

    def failures(self, k: int = 3) -> list[QuestionResult]:
        return [r for r in self.results if not r.hit_at(k)]

    def summary(self) -> dict:
        data = {
            "mode": self.mode,
            "top_k": self.top_k,
            "questions": self.n,
            "mrr": round(self.mrr(), 4),
        }
        for k in self.k_values:
            data[f"recall@{k}"] = round(self.recall_at(k), 4)
        return data

    def to_markdown(self, show_failures: int = 0) -> str:
        head = "| 指标 | 数值 |\n| --- | --- |\n"
        head += f"| 问题数 | {self.n} |\n"
        head += f"| MRR | {self.mrr():.4f} |\n"
        for k in self.k_values:
            head += f"| Recall@{k} | {self.recall_at(k):.1%} |\n"

        lines = [head]
        if show_failures:
            missed = self.failures(min(self.k_values))
            if missed:
                lines.append(f"\n未在 top-{min(self.k_values)} 命中的问题（{len(missed)} 条）：\n")
                for result in missed[:show_failures]:
                    want = " 或 ".join(str(e) for e in result.question.expect)
                    lines.append(f"- `{result.question.qid}` {result.question.question}")
                    lines.append(f"  - 期望：{want}")
                    lines.append(f"  - 实际：{' / '.join(result.top[:3]) or '（无结果）'}")
        return "\n".join(lines)


def load_questions(path: str | Path) -> list[Question]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在：{path}")
    questions: list[Question] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        try:
            questions.append(Question.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path}:{lineno} 解析失败：{exc}") from exc
    return questions


def evaluate(rag, questions: list[Question], *, mode: str = "bm25",
             top_k: int = 5, k_values: tuple[int, ...] = DEFAULT_K_VALUES) -> EvalReport:
    """跑评测。rag 只要有 search(query, top_k, mode) 即可。"""
    results: list[QuestionResult] = []
    for question in questions:
        hits = rag.search(question.question, top_k=top_k, mode=mode)
        ranks = [
            hit.rank
            for hit in hits
            if any(expectation.matches(hit.chunk) for expectation in question.expect)
        ]
        results.append(
            QuestionResult(
                question=question,
                ranks=ranks,
                top=[hit.label for hit in hits],
            )
        )
    return EvalReport(mode=mode, top_k=top_k, results=results, k_values=k_values)
