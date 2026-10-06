"""答案生成。

两种模式：
  1. extractive（默认，零依赖）：不做生成，直接把最相关的片段按相关性排好并给出出处。
  2. llm（可选）：把检索到的片段作为上下文，调用任意 OpenAI 兼容 /chat/completions 接口生成答案。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import LLMConfig
from .retriever import Hit

SYSTEM_PROMPT = """你是一名医疗设备（输液泵、注射泵）领域的技术助手。

回答规则：
1. 只依据【参考资料】作答，不要引入资料之外的事实，也不要凭经验补充。
2. 每一条结论后面用 [编号] 标注来源，编号对应参考资料的序号。
3. 如果参考资料不足以回答，直接说明"现有知识库中没有足够信息回答该问题"，并指出缺什么。
4. 涉及临床操作时，提醒以医院制度和设备说明书为准。
5. 用简洁的中文回答，必要时用列表或表格；先给结论，再给依据。"""

DISCLAIMER = (
    "> 以上内容由本地知识库检索生成，仅供学习与检索参考，"
    "不能替代设备说明书、院内操作规程与临床判断。"
)


def build_context(hits: list[Hit], max_chars: int = 7000, max_chunk_chars: int = 1400) -> str:
    """把检索结果拼成给大模型看的上下文。"""
    blocks: list[str] = []
    total = 0
    for hit in hits:
        text = hit.chunk.text
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars] + " …（已截断）"
        block = (
            f"[{hit.rank}] 出处：{hit.chunk.doc_title} › {hit.chunk.heading}"
            f"（文件：{hit.chunk.source}）\n{text}"
        )
        if total + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


def extractive_answer(query: str, hits: list[Hit], max_chunk_chars: int = 900) -> str:
    """无大模型时：返回最相关的原文片段。"""
    if not hits:
        return f"知识库中没有检索到与「{query}」相关的内容。可以换个说法，或先运行 build 重建索引。"

    lines = [
        f"**问题：** {query}",
        "",
        f"未配置大模型，以下是知识库中与问题最相关的 {len(hits)} 个片段（按相关度排序）：",
        "",
    ]
    for hit in hits:
        text = hit.chunk.text
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars].rstrip() + " …"
        lines.append(
            f"**[{hit.rank}] {hit.chunk.doc_title} › {hit.chunk.heading}**"
            f"  \n`{hit.chunk.source}` · 相关度 {hit.score:.4f}"
        )
        lines.append("")
        lines.append(text)
        lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def llm_answer(query: str, hits: list[Hit], cfg: LLMConfig) -> str:
    """调用 OpenAI 兼容接口生成答案。"""
    if not cfg.enabled:
        raise RuntimeError("大模型未配置：请设置 RAG_LLM_BASE_URL 与 RAG_LLM_MODEL。")
    if not hits:
        return f"知识库中没有检索到与「{query}」相关的内容。"

    context = build_context(hits)
    user_prompt = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        f"请依据参考资料作答，并用 [编号] 标注来源。"
    )

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
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
    except urllib.error.HTTPError as exc:  # pragma: no cover - 取决于外部服务
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"大模型接口返回 {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover
        raise RuntimeError(f"无法连接大模型接口：{exc.reason}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"大模型返回内容为空：{json.dumps(body, ensure_ascii=False)[:400]}")
    text = (choices[0].get("message") or {}).get("content", "").strip()
    return f"{text}\n\n{DISCLAIMER}" if text else "大模型没有返回内容。"
