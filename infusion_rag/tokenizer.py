"""分词与索引前文本规整。

中文不走分词器，而是用「单字 + 二元组」。理由是这里的语料以专业术语为主
（输液泵、蠕动泵、阻塞报警、气泡检测），字符二元组不需要词典就能召回，
也不会把 `mL/h`、`IEC 60601-2-24`、`0.1μg/kg/min` 这类单位切坏。

英文与数字按词切，并保留内部出现的 `.-/+%`，这样型号与标准号是一个整体 token。
"""
from __future__ import annotations

import re

# 拉丁词：字母数字开头，允许内部出现 . _ - / + %
_LATIN = r"[a-z0-9]+(?:[._\-/+%][a-z0-9]+)*"
# CJK 连续段（含扩展 A 区与兼容区）
_CJK = r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"

_TOKEN_RE = re.compile(rf"{_LATIN}|{_CJK}+")

# 索引前要剥掉的 Markdown 结构
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_CODE_RE = re.compile(r"`([^`]*)`")
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.*?)(?<=\S)\1")


def normalize(text: str) -> str:
    """把 Markdown 还原成纯文本，避免 URL 和标记符号进入索引。"""
    if not text:
        return ""
    text = _IMAGE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)   # 保留链接文字，丢掉 URL
    text = _CODE_RE.sub(r"\1", text)   # 保留行内代码内容
    text = _EMPHASIS_RE.sub(r"\2", text)
    return text


def tokenize(text: str, *, unigrams: bool = True, bigrams: bool = True) -> list[str]:
    """切词。默认同时产出一元和二元 token。"""
    if not text:
        return []
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        piece = match.group(0)
        if piece[0].isascii():
            tokens.append(piece)
            continue
        if unigrams:
            tokens.extend(piece)
        if bigrams:
            tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
    return tokens


def tokenize_for_index(text: str) -> list[str]:
    """建索引用：先规整 Markdown，再切词。"""
    return tokenize(normalize(text))


def unique_tokens(text: str) -> list[str]:
    """去重且保持顺序，用于查询扩展。"""
    seen: set[str] = set()
    out: list[str] = []
    for token in tokenize(normalize(text)):
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out
