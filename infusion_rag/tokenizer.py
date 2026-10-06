"""中英文混排分词：CJK 走"单字 + 二元组"，拉丁文走词 + 数字/单位。

不需要 jieba 之类的分词依赖：字符二元组（char bigram）在中文检索里表现稳定，
而且对"输液泵/注射泵/阻塞报警"这类专业词无需词典也能召回。
"""

from __future__ import annotations

import re

# 拉丁词（含数字、单位、连字符、斜杠、百分号），例如 ml/h、0.1μg、iec 60601-2-24
_LATIN = r"[a-z0-9]+(?:[._\-/+%][a-z0-9]+)*"
# CJK 连续段
_CJK = r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"

_TOKEN_RE = re.compile(rf"{_LATIN}|{_CJK}")

# 单字（CJK）权重低于二元组，避免噪声词主导
UNIGRAM = True


def tokenize(text: str) -> list[str]:
    """把一段文本切成检索用的 token 列表。"""
    if not text:
        return []
    text = text.lower()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        piece = match.group(0)
        if piece[0].isascii():
            tokens.append(piece)
        else:
            # CJK：单字 + 二元组
            if UNIGRAM:
                tokens.extend(piece)
            tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
    return tokens


def tokenize_unique(text: str) -> list[str]:
    """去重但保持顺序，用于查询扩展。"""
    seen = set()
    out = []
    for tok in tokenize(text):
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out
