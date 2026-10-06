"""把 Markdown 语料切成可检索的片段。

切分顺序：

1. 拆出前置元数据（出处、摘要、更新时间），正文里不再保留"参考来源"这类清单
   —— 出处是元数据而不是知识内容，放进索引只会让关键词在网址里虚高命中。
2. 按 `#` 标题切章节；章节里再按空行分段，超长段落按句末标点切。
3. Markdown 表格单独处理：按行切，每一块都补上表头，避免表格被切断后失去含义。
4. 相邻块之间保留一段重叠，防止答案正好落在切口上。

标题和正文分开存放：标题作为独立字段参与 BM25 打分（见 bm25.py 的字段化设计），
正文里不再重复写一遍标题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import ChunkConfig
from .frontmatter import DocMeta, Source, split_front_matter

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_TABLE_SEP_RE = re.compile(r"^\|[\s:\-|]+\|$")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*|(?<=\.)\s+")
_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")

# 这些标题下的内容视为出处清单，不进索引（出处由前置元数据承载）
_SOURCE_HEADING_RE = re.compile(
    r"(参考来源|资料来源|主要来源|出处|参考文献|references?|sources?)", re.I
)

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt")


@dataclass
class Chunk:
    """一个检索单元。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    path: str
    heading: str
    text: str
    order: int
    n_chars: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "path": self.path,
            "heading": self.heading,
            "text": self.text,
            "order": self.order,
            "n_chars": self.n_chars,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            doc_title=data["doc_title"],
            path=data.get("path", ""),
            heading=data.get("heading", ""),
            text=data.get("text", ""),
            order=int(data.get("order", 0)),
            n_chars=int(data.get("n_chars", len(data.get("text", "")))),
        )

    @property
    def label(self) -> str:
        """给人和给模型看的定位串。"""
        return f"{self.doc_title} › {self.heading}" if self.heading else self.doc_title


@dataclass
class Document:
    meta: DocMeta
    chunks: list[Chunk] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 文本切分
# --------------------------------------------------------------------------- #
def _overlap_tail(text: str, n: int) -> str:
    """取尾部 n 个字符作为重叠，尽量从段落或句末边界开始。"""
    if n <= 0 or len(text) <= n:
        return ""
    tail = text[-n:]
    for sep in ("\n\n", "\n", "。", "；", "，", " "):
        idx = tail.find(sep)
        if 0 <= idx < len(tail) - 8:
            return tail[idx + len(sep) :].strip()
    return tail.strip()


def _split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENT_SPLIT_RE.split(text)) if s]


def _split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """把超长段落按句子拆到 <= max_chars；单句仍过长就硬切。"""
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    if len(paragraph) <= max_chars:
        return [paragraph]

    out: list[str] = []
    current = ""
    for sentence in _split_sentences(paragraph):
        while len(sentence) > max_chars:
            head, sentence = sentence[:max_chars], sentence[max_chars:]
            if current:
                out.append(current)
                current = ""
            out.append(head)
        if current and len(current) + len(sentence) > max_chars:
            out.append(current)
            current = sentence
        else:
            current = sentence if not current else current + sentence
    if current:
        out.append(current)
    return out


def _split_table(table: str, max_chars: int) -> list[str]:
    """按行切表格，并给每一块补上表头。"""
    if len(table) <= max_chars:
        return [table]

    rows = [r for r in table.split("\n") if r.strip()]
    if len(rows) >= 2 and _TABLE_SEP_RE.match(rows[1].strip()):
        header, body = rows[:2], rows[2:]
    else:
        header, body = rows[:1], rows[1:]

    out: list[str] = []
    current = list(header)
    for row in body:
        if len("\n".join(current + [row])) > max_chars and len(current) > len(header):
            out.append("\n".join(current))
            current = list(header) + [row]
        else:
            current.append(row)
    if current:
        out.append("\n".join(current))
    return out


def _iter_blocks(text: str):
    """把正文切成 ("table" | "para", 块内容) 的序列。"""
    buffer: list[str] = []
    kind: str | None = None

    def flush():
        if buffer:
            block = "\n".join(buffer).strip()
            if block:
                yield (kind, block)

    for line in text.split("\n"):
        line_kind = "table" if line.lstrip().startswith("|") else "para"
        if kind is None:
            kind = line_kind
        if line_kind != kind:
            yield from flush()
            buffer = []
            kind = line_kind
        buffer.append(line)
    yield from flush()


def split_section(text: str, cfg: ChunkConfig) -> list[str]:
    """把一个章节的正文切成若干片段。"""
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
            pieces.extend(_split_paragraph(block, cfg.max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if current and len(current) + len(piece) + 2 > cfg.max_chars:
            chunks.append(current)
            tail = _overlap_tail(current, cfg.overlap_chars)
            current = f"{tail}\n\n{piece}" if tail else piece
        else:
            current = piece if not current else f"{current}\n\n{piece}"
    if current:
        chunks.append(current)

    if len(chunks) >= 2 and len(chunks[-1]) < cfg.min_chars:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks[-1]}"
        chunks.pop()
    return chunks


# --------------------------------------------------------------------------- #
# 文档解析
# --------------------------------------------------------------------------- #
def _harvest_sources(text: str) -> list[Source]:
    """从"参考来源"小节里捡出 Markdown 链接，转成结构化出处。"""
    found: list[Source] = []
    seen: set[str] = set()
    for label, url in _LINK_RE.findall(text):
        key = url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        found.append(Source(label=label.strip(), url=url.strip()))
    return found


def _sections(body: str) -> tuple[str, list[tuple[str, str]]]:
    """按标题把正文切成分节。返回 (H1 标题, [(章节路径, 正文)])。"""
    h1 = ""
    stack: list[tuple[int, str]] = []
    result: list[tuple[str, str]] = []
    heading = ""
    lines: list[str] = []
    started = False

    def flush():
        content = "\n".join(lines).strip()
        if content:
            result.append((heading, content))

    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if not match:
            lines.append(line)
            continue

        if started:
            flush()
        else:
            started = True
        lines = []

        level = len(match.group(1))
        title = match.group(2).strip()
        if level == 1 and not h1:
            h1 = title
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        # 章节路径只取二级及以下标题，H1 已经作为文档标题
        parts = [t for lvl, t in stack if lvl >= 2]
        heading = " › ".join(parts)

    flush()
    return h1, result


def parse_document(path: Path, corpus_root: Path, cfg: ChunkConfig) -> Document:
    """读一个文件，返回它的元数据和片段。"""
    raw = path.read_text(encoding="utf-8")
    meta_dict, body = split_front_matter(raw)

    doc_id = path.stem
    relative = path.relative_to(corpus_root).as_posix()

    h1, sections = _sections(body)

    harvested: list[Source] = []
    kept: list[tuple[str, str]] = []
    for heading, content in sections:
        if _SOURCE_HEADING_RE.search(heading.split(" › ")[-1]):
            harvested.extend(_harvest_sources(content))
            continue
        kept.append((heading, content))

    raw_sources = meta_dict.get("sources") or []
    sources: list[Source] = []
    for item in raw_sources:
        if isinstance(item, dict):
            sources.append(Source.from_dict(item))
        elif isinstance(item, str):
            sources.append(Source(label=item))
    for source in harvested:
        if all(s.url != source.url for s in sources):
            sources.append(source)

    meta = DocMeta(
        doc_id=doc_id,
        title=str(meta_dict.get("title") or h1 or doc_id),
        summary=str(meta_dict.get("summary") or ""),
        updated=str(meta_dict.get("updated") or ""),
        path=relative,
        tags=[str(t) for t in (meta_dict.get("tags") or [])],
        sources=sources,
    )

    chunks: list[Chunk] = []
    order = 0
    for heading, content in kept:
        for piece in split_section(content, cfg):
            order += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{order:03d}",
                    doc_id=doc_id,
                    doc_title=meta.title,
                    path=relative,
                    heading=heading,
                    text=piece,
                    order=order,
                    n_chars=len(piece),
                )
            )

    meta.n_chunks = len(chunks)
    meta.n_chars = sum(c.n_chars for c in chunks)
    return Document(meta=meta, chunks=chunks)


def load_corpus(
    corpus_dir: str | Path,
    cfg: ChunkConfig | None = None,
    suffixes: tuple[str, ...] = SUPPORTED_SUFFIXES,
) -> list[Document]:
    """遍历语料目录，按文件名排序返回所有文档。"""
    cfg = cfg or ChunkConfig()
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"语料目录不存在：{root}")

    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    return [parse_document(path, root, cfg) for path in files]
