"""Markdown 分块：按标题切章节，章节内按段落/句子滑动窗口切块，表格按行切并保留表头。"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .config import ChunkConfig

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_TABLE_SEP_RE = re.compile(r"^\|[\s:\-|]+\|$")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*|(?<=\.)\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source: str
    heading: str
    text: str
    order: int
    n_chars: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


# --------------------------------------------------------------------------- #
# 文本切分
# --------------------------------------------------------------------------- #
def _tail(text: str, n: int) -> str:
    """取尾部 n 个字符作为重叠，尽量从段落/句子边界开始。"""
    if n <= 0 or len(text) <= n:
        return ""
    tail = text[-n:]
    for sep in ("\n\n", "\n", "。", "；", "，", " "):
        idx = tail.find(sep)
        if 0 <= idx < len(tail) - 8:
            return tail[idx + len(sep) :].strip()
    return tail.strip()


def _split_sentences(text: str) -> list[str]:
    return [s for s in (piece.strip() for piece in _SENT_SPLIT_RE.split(text)) if s]


def _hard_split(paragraph: str, max_chars: int) -> list[str]:
    """把超长段落按句子拆到 <= max_chars，单句过长则硬切。"""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    if len(paragraph) <= max_chars:
        return [paragraph]

    out: list[str] = []
    cur = ""
    for sentence in _split_sentences(paragraph):
        while len(sentence) > max_chars:
            head, sentence = sentence[:max_chars], sentence[max_chars:]
            if cur:
                out.append(cur)
                cur = ""
            out.append(head)
        if cur and len(cur) + len(sentence) > max_chars:
            out.append(cur)
            cur = sentence
        else:
            cur = sentence if not cur else cur + sentence
    if cur:
        out.append(cur)
    return out


def _split_table(table: str, max_chars: int) -> list[str]:
    """按行切表格，每块都带上表头，避免脱离表头后无法理解。"""
    if len(table) <= max_chars:
        return [table]

    rows = [r for r in table.split("\n") if r.strip()]
    if len(rows) >= 2 and _TABLE_SEP_RE.match(rows[1].strip()):
        header, body = rows[:2], rows[2:]
    else:
        header, body = rows[:1], rows[1:]

    out: list[str] = []
    cur = list(header)
    for row in body:
        candidate = "\n".join(cur + [row])
        if len(candidate) > max_chars and len(cur) > len(header):
            out.append("\n".join(cur))
            cur = list(header) + [row]
        else:
            cur.append(row)
    if cur:
        out.append("\n".join(cur))
    return out


def _iter_blocks(text: str):
    """把文本切成 ("table"|"para", block) 序列。"""
    buf: list[str] = []
    kind: str | None = None

    def flush():
        if buf:
            block = "\n".join(buf).strip()
            if block:
                yield (kind, block)

    for line in text.split("\n"):
        line_kind = "table" if line.lstrip().startswith("|") else "para"
        if kind is None:
            kind = line_kind
        if line_kind != kind:
            yield from flush()
            buf = []
            kind = line_kind
        buf.append(line)
    yield from flush()


def split_text(text: str, cfg: ChunkConfig) -> list[str]:
    """把一段章节正文切成若干块。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= cfg.max_chars:
        return [text]

    pieces: list[str] = []
    for kind, block in _iter_blocks(text):
        if kind == "table":
            pieces.extend(_split_table(block, cfg.max_chars))
        else:
            pieces.extend(_hard_split(block, cfg.max_chars))

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if cur and len(cur) + len(piece) + 2 > cfg.max_chars:
            chunks.append(cur)
            overlap = _tail(cur, cfg.overlap_chars)
            cur = f"{overlap}\n\n{piece}" if overlap else piece
        else:
            cur = piece if not cur else f"{cur}\n\n{piece}"
    if cur:
        chunks.append(cur)

    # 合并过短的尾块
    if len(chunks) >= 2 and len(chunks[-1]) < cfg.min_chars:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
        chunks.pop()

    return chunks


# --------------------------------------------------------------------------- #
# 文件 / 语料切分
# --------------------------------------------------------------------------- #
def chunk_markdown(text: str, *, doc_id: str, doc_title: str, source: str,
                   cfg: ChunkConfig, start_index: int = 0) -> list[Chunk]:
    """把一篇 Markdown 文本切成 Chunk 列表。"""
    sections: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    current_heading = ""
    body: list[str] = []
    title = doc_title
    title_fixed = False

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            sections.append((current_heading, body))
            body = []
            level = len(match.group(1))
            heading = match.group(2).strip()
            if level == 1 and not title_fixed:
                title = heading or title
                title_fixed = True
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            parts = [t for lvl, t in stack if lvl >= 2]
            current_heading = " › ".join(parts) if parts else ""
        else:
            body.append(line)
    sections.append((current_heading, body))

    chunks: list[Chunk] = []
    order = start_index
    for heading, lines in sections:
        raw = "\n".join(lines).strip()
        if not raw:
            continue
        prefix = f"### {heading}\n\n" if heading else ""
        for piece in split_text(raw, cfg):
            order += 1
            body_text = piece.strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{order:03d}",
                    doc_id=doc_id,
                    doc_title=title,
                    source=source,
                    heading=heading,
                    text=f"{prefix}{body_text}".strip(),
                    order=order,
                    n_chars=len(body_text),
                )
            )
    return chunks


def chunk_file(path: Path, *, corpus_root: Path, cfg: ChunkConfig) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(corpus_root).as_posix()
    doc_id = path.stem
    return chunk_markdown(
        text,
        doc_id=doc_id,
        doc_title=path.stem,
        source=rel,
        cfg=cfg,
    )


def chunk_corpus(corpus_dir: str | Path, cfg: ChunkConfig | None = None,
                 patterns: tuple[str, ...] = ("*.md", "*.markdown", "*.txt")) -> list[Chunk]:
    """遍历语料目录，返回全部 Chunk。"""
    cfg = cfg or ChunkConfig()
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"语料目录不存在：{root}")

    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    files = sorted(set(files), key=lambda p: p.as_posix())

    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_file(path, corpus_root=root, cfg=cfg))
    return chunks
