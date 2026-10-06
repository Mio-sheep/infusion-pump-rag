"""答案生成。

两种方式：

  extractive  不调模型，把最相关的原文片段按相关度排好并标注出处。
              零成本、零幻觉，对"我要找原文"这类用法完全够用。
  llm         把检索到的片段作为唯一上下文交给大模型，要求逐条标注 [编号]。
              接入任何 OpenAI 兼容的 /chat/completions 接口。

无论哪种方式，出处都取自索引里的文档元数据，而不是让模型自己编。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import LLMConfig
from .retriever import Hit

SYSTEM_PROMPT = """你是医用输液泵与注射泵领域的技术资料助手。

回答规则：
1. 只依据【资料】作答。资料里没有的内容，直接说"知识库中没有相关内容"，不要用先验知识补充。
2. 每条结论后面标注来源编号，写成 [1]、[2] 这种形式。
3. 资料之间如果互相矛盾，把矛盾点讲出来，不要替它们调和。
4. 涉及临床操作、设备维修或计量校准的结论，提醒以设备说明书、现行标准和本院制度为准。
5. 先给结论，再给依据。用中文，语言简洁，不要用"首先/其次/最后"这类套话。"""

FOOTER = (
    "---\n"
    "*以上内容由本地知识库检索得到，仅供查资料用。"
    "实际处置请以设备说明书、现行标准文本和本院规章制度为准。*"
)


def build_context(hits: list[Hit], max_chars: int = 7000, max_chunk_chars: int = 1400) -> str:
    """拼出给大模型的上下文。每块带定位信息，方便模型写 [编号]。"""
    blocks: list[str] = []
    used = 0
    for hit in hits:
        text = hit.chunk.text
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars].rstrip() + " …（原文较长，此处截断）"
        block = f"[{hit.rank}] 位置：{hit.label}\n{text}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def format_sources(hits: list[Hit], limit: int = 6) -> str:
    """把命中片段所属文档的出处去重列出。"""
    lines: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        for source in hit.sources:
            label = source.get("label", "")
            url = source.get("url", "")
            key = url or label
            if not key or key in seen:
                continue
            seen.add(key)
            lines.append(f"- [{label}]({url})" if url else f"- {label}")
            if len(lines) >= limit:
                return "\n".join(lines)
    return "\n".join(lines)


def extractive_answer(query: str, hits: list[Hit], max_chunk_chars: int = 900) -> str:
    """不调模型：给出最相关的原文片段。"""
    if not hits:
        return (
            f"知识库中没有检索到与「{query}」相关的内容。\n\n"
            f"可以换个说法再试，或用 `python -m infusion_rag.cli search` 看看命中了什么。"
        )

    parts = [
        f"**问题：** {query}",
        "",
        f"未配置大模型，下面按相关度列出知识库中最相关的 {len(hits)} 个片段。",
        "",
    ]
    for hit in hits:
        text = hit.chunk.text
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars].rstrip() + " …"
        parts.append(f"**[{hit.rank}] {hit.label}**  ")
        parts.append(f"`{hit.chunk.path}` · 相关度 {hit.score:.4f}")
        parts.append("")
        parts.append(text)
        parts.append("")

    sources = format_sources(hits)
    if sources:
        parts.extend(["**出处：**", sources, ""])
    parts.append(FOOTER)
    return "\n".join(parts)


def llm_answer(query: str, hits: list[Hit], cfg: LLMConfig) -> str:
    """调用 OpenAI 兼容接口生成答案。"""
    if not cfg.enabled:
        raise RuntimeError("大模型未配置：请设置 RAG_LLM_BASE_URL 与 RAG_LLM_MODEL。")
    if not hits:
        return f"知识库中没有检索到与「{query}」相关的内容。"

    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"【资料】\n{build_context(hits)}\n\n"
                    f"【问题】\n{query}\n\n"
                    f"请依据资料作答，并用 [编号] 标注每一条结论的来源。"
                ),
            },
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }

    request = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - 依赖外部服务
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"大模型接口返回 {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - 依赖外部服务
        raise RuntimeError(f"无法连接大模型接口：{exc.reason}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"大模型返回内容为空：{json.dumps(body, ensure_ascii=False)[:400]}")
    text = (choices[0].get("message") or {}).get("content", "").strip()
    if not text:
        return "大模型没有返回内容。"

    sources = format_sources(hits)
    tail = f"\n\n**出处：**\n{sources}\n" if sources else "\n"
    return f"{text}{tail}\n{FOOTER}"
