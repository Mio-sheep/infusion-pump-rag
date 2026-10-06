"""文档前置元数据（front matter）解析。

知识库里的每篇 Markdown 都以一段 YAML 风格的头信息开头：

    ---
    title: 报警与故障排查
    summary: 六类常见报警的可能原因与分级处置
    updated: 2026-02-28
    sources:
      - label: FDA, Examples of Reported Infusion Pump Problems
        url: https://www.fda.gov/...
      - label: JJF 1259-2010 医用注射泵和输液泵校准规范
        url: https://www.ndls.org.cn/...
    ---

只支持这个项目实际用到的 YAML 子集（标量、字符串列表、对象列表），
不引入 PyYAML —— 整个项目保持零第三方依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field

FENCE = "---"


@dataclass
class Source:
    """一条出处。label 是给人看的，url 是可点击的（可能为空）。"""

    label: str
    url: str = ""

    def to_dict(self) -> dict:
        return {"label": self.label, "url": self.url}

    @classmethod
    def from_dict(cls, data: dict) -> "Source":
        return cls(label=str(data.get("label", "")), url=str(data.get("url", "")))


@dataclass
class DocMeta:
    """一篇文档的元数据。sources 会挂在检索结果上一并返回，方便标注出处。"""

    doc_id: str
    title: str
    summary: str = ""
    updated: str = ""
    path: str = ""
    tags: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    n_chunks: int = 0
    n_chars: int = 0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "summary": self.summary,
            "updated": self.updated,
            "path": self.path,
            "tags": list(self.tags),
            "sources": [s.to_dict() for s in self.sources],
            "n_chunks": self.n_chunks,
            "n_chars": self.n_chars,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocMeta":
        return cls(
            doc_id=data["doc_id"],
            title=data.get("title", data["doc_id"]),
            summary=data.get("summary", ""),
            updated=data.get("updated", ""),
            path=data.get("path", ""),
            tags=list(data.get("tags", [])),
            sources=[Source.from_dict(s) for s in data.get("sources", [])],
            n_chunks=int(data.get("n_chunks", 0)),
            n_chars=int(data.get("n_chars", 0)),
        )


def split_front_matter(text: str) -> tuple[dict, str]:
    """拆出前置元数据。

    返回 (元数据字典, 正文)。没有前置元数据时返回 ({}, 原文)。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FENCE:
        return {}, text

    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == FENCE:
            end = i
            break
    if end is None:  # 只有开头一个 ---，当作正文
        return {}, text

    meta = _parse_block(lines[1:end])
    body = "\n".join(lines[end + 1 :])
    return meta, body


def _parse_block(lines: list[str]) -> dict:
    """解析 YAML 子集：`key: value` 与 `key:` 加缩进的 `- item` 列表。"""
    meta: dict = {}
    current_key: str | None = None
    current_item: dict | None = None

    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue

        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()

        # 列表项
        if line.startswith("- "):
            value = line[2:].strip()
            if current_key is None:
                continue
            bucket = meta.setdefault(current_key, [])
            if not isinstance(bucket, list):
                continue
            if ":" in value and not value.startswith(("http://", "https://")):
                # 对象列表的第一行，例如 "- label: xxx"
                key, _, val = value.partition(":")
                current_item = {key.strip(): _scalar(val)}
                bucket.append(current_item)
            else:
                current_item = None
                bucket.append(_scalar(value))
            continue

        # 对象列表的续行（缩进更深，形如 "  url: https://..."）
        if current_item is not None and indent > 0 and ":" in line:
            key, _, val = line.partition(":")
            current_item[key.strip()] = _scalar(val)
            continue

        # 顶层 key
        if indent == 0 and ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            current_item = None
            if val:
                meta[key] = _scalar(val)
                current_key = None
            else:
                current_key = key
                meta.setdefault(key, [])

    return meta


def _scalar(value: str):
    """去掉引号，识别常见的字面量。"""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~", ""}:
        return ""
    return value
