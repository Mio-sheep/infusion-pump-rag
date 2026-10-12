"""把 PDF 语料接入现有的切块链路。

设计：PDF → 伪 Markdown → chunker.py 原有流程，分块、重叠、字段化打分全部复用。

为什么这样做，而不是对 PDF 单独写一套切块：

1. 下游（BM25、检索、评测）不关心语料从哪来，多一条特殊路径就多一分
   不一致的可能。Markdown 是这条流水线内部的"通用中间格式"。
2. PDF 的结构信息（标题层级）没有显式标记，需要从**字号**推断：
   正文用出现最多的字号，明显更大的字号按大小档位映射到标题层级。
   这一步只在转换时做一次。

已知边界（诚实列出）：

- 扫描版 PDF（图片、无文字层）提取不到任何文字，会抛出明确的错误提示
  需要 OCR，本工具不做 OCR。
- PDF 表格还原不可靠：提取器只给文字流，列关系会丢。表格建议整理成
  Markdown 再放进语料。
- 复杂排版（多栏、页眉页脚、脚注）可能把不相关的行拼进同一段落。
  换语料后请用 `python -m infusion_rag.cli search` 抽查几个查询。

依赖：pypdf（可选，纯 Python）。核心 BM25 链路依然是零依赖，
没有 PDF 语料就不需要装它。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

# 字号聚类和标题判断的阈值，单位是 PDF 点（pt）
_Y_TOLERANCE = 2.5  # y 坐标相差不超过这个值算同一行
_HEADING_MARGIN = 1.5  # 比正文大这么多才算标题
_MAX_HEADING_LEVEL = 4  # 最大的字号档映射到 # ，往下依次 ## ### ####


class PdfError(Exception):
    """PDF 处理不了时抛出，消息直接写给用户看。"""


def _require_pypdf():
    try:
        from pypdf import PdfReader  # noqa: F401
    except ImportError as exc:
        raise PdfError(
            "处理 PDF 需要可选依赖 pypdf：\n"
            "    pip install pypdf\n"
            "装好之后重新运行即可。不需要 PDF 语料的话，把 data/raw/ 里的 "
            ".pdf 文件移走就行。"
        ) from exc
    from pypdf import PdfReader

    return PdfReader


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or ch in "，。；：、！？（）"


def _join_lines(a: str, b: str) -> str:
    """拼两行视觉文本：中文之间直接接，拉丁文之间补空格。"""
    if not a or not b:
        return a + b
    if _is_cjk(a[-1]) or _is_cjk(b[:1]):
        return a + b
    return a + " " + b


def _page_lines(page) -> list[tuple[str, float]]:
    """提取一页，返回 [(行文本, 该行最大字号)]，自上而下。"""
    frags: list[tuple[str, float, float, float]] = []  # text, x, y, size

    def visitor(text, cm, tm, font_dict, font_size):  # noqa: ARG001
        if text and text.strip():
            frags.append((text, tm[4], tm[5], float(font_size or 0)))

    page.extract_text(visitor_text=visitor)
    if not frags:
        return []

    # 按 y 聚类成行（PDF 的 y 向上增长，行号取负保证自上而下的顺序）
    rows: list[list[tuple[float, float, str, float]]] = []  # [-y, x, text, size]
    for text, x, y, size in frags:
        key = -round(y / _Y_TOLERANCE)
        for row in rows:
            if abs(row[0][0] - key) <= 1:
                row.append((key, x, text, size))
                break
        else:
            rows.append([(key, x, text, size)])

    lines: list[tuple[str, float]] = []
    for row in sorted(rows, key=lambda r: r[0][0]):
        parts = sorted(row, key=lambda p: p[1])  # 按横坐标排序
        line = ""
        for _, _, text, _ in parts:
            line = _join_lines(line, text)
        line = line.strip()
        if line:
            lines.append((line, max(p[3] for p in parts)))
    return lines


def _body_size(lines: list[tuple[str, float]]) -> float:
    """正文字号 = 按字符数加权出现最多的字号。"""
    counter: Counter[float] = Counter()
    for text, size in lines:
        counter[round(size * 2) / 2] += len(text)
    return counter.most_common(1)[0][0] if counter else 0.0


def _heading_levels(lines: list[tuple[str, float]], body: float) -> dict[float, int]:
    """把比正文大的字号按档位映射到标题层级（大 → 浅）。"""
    bigger = sorted(
        {round(size * 2) / 2 for _, size in lines if size >= body + _HEADING_MARGIN},
        reverse=True,
    )
    return {size: idx + 1 for idx, size in enumerate(bigger[:_MAX_HEADING_LEVEL])}


def _render(pages: list[list[tuple[str, float]]]) -> tuple[str, str]:
    """把所有页渲染成伪 Markdown。返回 (H1 候选标题, 正文)。"""
    all_lines = [line for page in pages for line in page]
    body = _body_size(all_lines)
    levels = _heading_levels(all_lines, body)

    out: list[str] = []
    para: list[str] = []
    h1 = ""

    def flush():
        if para:
            merged = para[0]
            for line in para[1:]:
                merged = _join_lines(merged, line)
            out.append(merged)
            para.clear()

    for page in pages:
        for line, size in page:
            level = levels.get(round(size * 2) / 2)
            if level:
                flush()
                out.append(f"{'#' * level} {line}")
                if level == 1 and not h1:
                    h1 = line
            else:
                para.append(line)
        flush()  # 页边界结束当前段落
    return h1, "\n\n".join(out)


def pdf_to_markdown(path: Path) -> str:
    """读一个 PDF，返回伪 Markdown 文本（喂给 chunker 的中间格式）。"""
    PdfReader = _require_pypdf()
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf 的异常类型不稳定，统一转成自己的
        raise PdfError(f"PDF 打不开：{path.name}（{exc}）") from exc

    pages: list[list[tuple[str, float]]] = []
    for page in reader.pages:
        try:
            pages.append(_page_lines(page))
        except PdfError:
            raise
        except Exception as exc:
            raise PdfError(f"PDF 第 {len(pages) + 1} 页解析失败：{exc}") from exc

    total = sum(len(page) for page in pages)
    if total == 0:
        raise PdfError(
            f"这个 PDF 一行文字都提取不到：{path.name}\n"
            "多半是扫描版（整页图片、没有文字层）。处理它需要 OCR，"
            "本工具不支持；请先用 OCR 工具转成文字版，或整理成 Markdown 放进语料。"
        )

    h1, body_md = _render(pages)

    # PDF 元数据里的标题比从字号猜的 H1 更可靠，有就优先用
    title = ""
    try:
        title = (reader.metadata or {}).get("/Title", "") or ""
    except Exception:
        title = ""

    header = ""
    if title or h1:
        header = f"---\ntitle: {title or h1}\n---\n\n"
    return header + body_md
