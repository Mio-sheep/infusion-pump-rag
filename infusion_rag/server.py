"""网页版检索界面。

用标准库 http.server，不引入 Flask / FastAPI —— 这个项目的依赖承诺是"零第三方包"，
为了一个单页界面破例不值得。

    python -m infusion_rag.cli serve --port 8000

接口：
    GET /                      单页界面
    GET /api/stats             索引统计
    GET /api/docs              语料文档与出处
    GET /api/search?q=&k=&mode=
    GET /api/ask?q=&k=&mode=&llm=auto|0|1
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import LLMConfig
from .pipeline import InfusionPumpRAG
from .retriever import MODES

MODE_LABELS = {
    "bm25": "关键词（BM25）",
    "dense": "语义（向量）",
    "hybrid": "混合（RRF 融合）",
}

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>输液泵 / 注射泵 知识库</title>
<style>
  :root {
    --bg: #fbfbfa; --panel: #ffffff; --line: #e3e2de; --line-strong: #cfcec9;
    --fg: #1f1e1c; --muted: #6b6a66; --accent: #1a5fb4; --accent-soft: #eaf1fb;
    --mono: ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI",
          "Microsoft YaHei", "PingFang SC", sans-serif;
  }
  .wrap { max-width: 820px; margin: 0 auto; padding: 0 20px; }
  header { padding: 40px 0 24px; border-bottom: 1px solid var(--line); }
  h1 { font-size: 21px; margin: 0 0 6px; font-weight: 650; letter-spacing: .2px; }
  .stats { color: var(--muted); font-size: 13px; font-family: var(--mono); }
  form.search { display: flex; gap: 8px; padding: 22px 0 14px; }
  input[type=text] {
    flex: 1; padding: 11px 14px; border: 1px solid var(--line-strong);
    border-radius: 6px; background: var(--panel); color: var(--fg);
    font: inherit; font-size: 15px; outline: none;
  }
  input[type=text]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
  button {
    padding: 11px 18px; border: 1px solid var(--accent); border-radius: 6px;
    background: var(--accent); color: #fff; font: inherit; font-size: 14px;
    cursor: pointer;
  }
  button:hover { background: #14509c; }
  button.secondary { background: var(--panel); color: var(--fg); border-color: var(--line-strong); }
  button.secondary:hover { background: #f2f1ee; }
  .controls { display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
              color: var(--muted); font-size: 13px; padding-bottom: 18px; }
  .controls label { display: flex; align-items: center; gap: 6px; }
  select { font: inherit; font-size: 13px; padding: 4px 6px; border-radius: 5px;
           border: 1px solid var(--line-strong); background: var(--panel); color: var(--fg); }
  .examples { display: flex; flex-wrap: wrap; gap: 8px; padding-bottom: 26px; }
  .examples button {
    background: transparent; border: 1px solid var(--line-strong); color: var(--muted);
    font-size: 12.5px; padding: 5px 11px; border-radius: 999px;
  }
  .examples button:hover { color: var(--fg); border-color: var(--accent); background: var(--panel); }
  main { padding-bottom: 72px; }
  .status { color: var(--muted); font-size: 13px; padding: 8px 0 18px; }
  .card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
    padding: 16px 18px; margin-bottom: 12px;
  }
  .card h3 { margin: 0 0 6px; font-size: 15px; font-weight: 620; line-height: 1.45; }
  .rank { color: var(--muted); font-family: var(--mono); font-size: 13px; margin-right: 8px; }
  .meta { color: var(--muted); font-size: 12px; font-family: var(--mono);
          margin-bottom: 12px; word-break: break-all; }
  pre { margin: 0; white-space: pre-wrap; word-break: break-word;
        font: inherit; font-size: 14px; }
  code { font-family: var(--mono); font-size: 13px; background: #f2f1ee;
         padding: 1px 5px; border-radius: 4px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0; }
  th, td { border: 1px solid var(--line); padding: 6px 9px; text-align: left; vertical-align: top; }
  th { background: #f6f5f2; font-weight: 600; }
  .sources { margin-top: 12px; padding-top: 10px; border-top: 1px dashed var(--line);
             font-size: 12.5px; color: var(--muted); }
  .sources a { color: var(--accent); }
  .answer { border-color: var(--accent); background: #f7faff; }
  .answer h3 { color: var(--accent); }
  .empty { color: var(--muted); font-size: 14px; }
  footer { border-top: 1px solid var(--line); color: var(--muted);
           font-size: 12.5px; padding: 20px 0 48px; line-height: 1.8; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>输液泵 / 注射泵 知识库</h1>
    <div class="stats" id="stats">正在读取索引…</div>
  </header>

  <form class="search" id="form">
    <input type="text" id="q" autocomplete="off" autofocus
           placeholder="例如：阻塞报警是什么原因 / JJF 1259 / occlusion bolus">
    <button type="submit" id="searchBtn">检索</button>
    <button type="button" class="secondary" id="askBtn">问答</button>
  </form>

  <div class="controls">
    <label>模式
      <select id="mode">__MODE_OPTIONS__</select>
    </label>
    <label>条数
      <select id="k">
        <option>3</option><option selected>4</option>
        <option>6</option><option>8</option>
      </select>
    </label>
    <span id="genMode"></span>
  </div>

  <div class="examples" id="examples"></div>
  <main id="out"></main>

  <footer>
    知识库内容整理自 FDA 输液泵公开文档、JJF 1259-2010、IEC 60601-2-24 等公开资料，
    供查资料使用，<b>不能替代设备说明书、现行标准文本和本院规章制度</b>。
  </footer>
</div>

<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// 只做最小的 Markdown 渲染：粗体、行内代码、链接。够用，且不会误伤原文。
function render(text) {
  let html = esc(text);
  html = html.replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^)\\s]+)\\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\\*\\*([^*]+)\\*\\*/g, '<b>$1</b>');
  return html;
}

function sourcesHtml(sources) {
  if (!sources || !sources.length) return '';
  const items = sources.map((s) =>
    s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.label)}</a>`
          : esc(s.label));
  return `<div class="sources">出处：${items.join('　·　')}</div>`;
}

function hitsHtml(hits) {
  if (!hits.length) {
    return '<div class="card empty">没有命中。换个说法，或用关键词（BM25）模式试试。</div>';
  }
  return hits.map((h) => `
    <div class="card">
      <h3><span class="rank">[${h.rank}]</span>${esc(h.doc_title)}${h.heading ? ' › ' + esc(h.heading) : ''}</h3>
      <div class="meta">${esc(h.path)} · ${esc(h.chunk_id)} · score ${h.score.toFixed(4)} · bm25 ${h.bm25.toFixed(3)} · dense ${h.dense.toFixed(3)}</div>
      <pre>${render(h.text)}</pre>
      ${sourcesHtml(h.sources)}
    </div>`).join('');
}

async function getJSON(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

async function doSearch() {
  const q = $('q').value.trim();
  if (!q) return;
  const mode = $('mode').value, k = $('k').value;
  $('out').innerHTML = '<div class="status">检索中…</div>';
  try {
    const d = await getJSON(`/api/search?q=${encodeURIComponent(q)}&k=${k}&mode=${mode}`);
    $('out').innerHTML =
      `<div class="status">${d.n_hits} 条结果 · 模式 ${esc(d.mode)}</div>` + hitsHtml(d.hits);
  } catch (e) {
    $('out').innerHTML = `<div class="card empty">出错：${esc(e.message)}</div>`;
  }
}

async function doAsk() {
  const q = $('q').value.trim();
  if (!q) return;
  const mode = $('mode').value, k = $('k').value;
  $('out').innerHTML = '<div class="status">生成中…</div>';
  try {
    const d = await getJSON(`/api/ask?q=${encodeURIComponent(q)}&k=${k}&mode=${mode}`);
    const answer = `<div class="card answer">
        <h3>回答 · ${esc(d.generator === 'llm' ? '大模型生成' : '抽取式（未配置大模型）')}</h3>
        <pre>${render(d.answer)}</pre>
      </div>`;
    $('out').innerHTML = answer +
      `<div class="status">以下是支撑答案的原文片段</div>` + hitsHtml(d.hits);
  } catch (e) {
    $('out').innerHTML = `<div class="card empty">出错：${esc(e.message)}</div>`;
  }
}

const EXAMPLES = [
  '阻塞报警怎么排查',
  '气泡报警可能是什么原因',
  '独立双人核对的定义',
  'JJF 1259 校准规范的要求',
  '按键连击会造成什么后果',
  '电池失效导致泵停机',
];

$('examples').innerHTML = EXAMPLES.map((e) =>
  `<button type="button" data-q="${esc(e)}">${esc(e)}</button>`).join('');
$('examples').addEventListener('click', (ev) => {
  const btn = ev.target.closest('button[data-q]');
  if (!btn) return;
  $('q').value = btn.dataset.q;
  doSearch();
});

$('form').addEventListener('submit', (ev) => { ev.preventDefault(); doSearch(); });
$('askBtn').addEventListener('click', doAsk);

getJSON('/api/stats').then((s) => {
  const parts = [
    `${s.n_documents} 篇文档`,
    `${s.n_chunks} 个片段`,
    `${s.n_sections} 个小节`,
    `词表 ${s.vocab_size}`,
    `平均 ${s.avg_chunk_chars} 字/片段`,
  ];
  $('stats').textContent = parts.join(' · ');
  $('genMode').textContent = s.llm_enabled
    ? `生成：大模型（${s.llm_model}）`
    : '生成：抽取式（未配置大模型）';
}).catch((e) => { $('stats').textContent = '统计信息读取失败：' + e.message; });
</script>
</body>
</html>
"""


def _make_handler(rag: InfusionPumpRAG, page: bytes, quiet: bool = False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "infusion-pump-rag"
        protocol_version = "HTTP/1.1"

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def log_message(self, fmt, *args):  # noqa: D102
            if not quiet:
                print(f"  {self.address_string()} {fmt % args}")

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            path = parsed.path.rstrip("/") or "/"

            if path == "/":
                self._send(page, "text/html; charset=utf-8")
                return

            if path == "/api/stats":
                stats = rag.stats()
                llm = LLMConfig.from_env()
                stats["llm_enabled"] = llm.enabled
                stats["llm_model"] = llm.model
                self._json(stats)
                return

            if path == "/api/docs":
                self._json({"documents": rag.documents()})
                return

            if path in {"/api/search", "/api/ask"}:
                self._handle_query(path, params)
                return

            self._json({"error": "404 not found"}, 404)

        def _handle_query(self, path: str, params: dict) -> None:
            query = (params.get("q") or [""])[0].strip()
            if not query:
                self._json({"error": "缺少查询参数 q"}, 400)
                return

            try:
                top_k = max(1, min(int((params.get("k") or ["4"])[0]), 20))
            except ValueError:
                top_k = 4

            mode = (params.get("mode") or ["bm25"])[0]
            if mode not in MODES:
                self._json({"error": f"未知模式 {mode}，可选：{', '.join(MODES)}"}, 400)
                return

            try:
                if path == "/api/search":
                    hits = rag.search(query, top_k=top_k, mode=mode)
                    self._json({
                        "query": query,
                        "mode": mode,
                        "n_hits": len(hits),
                        "hits": [h.to_dict() for h in hits],
                    })
                    return

                flag = (params.get("llm") or ["auto"])[0]
                use_llm = {"0": False, "1": True}.get(flag)
                self._json(rag.ask(query, top_k=top_k, mode=mode, use_llm=use_llm))
            except Exception as exc:  # pragma: no cover - 面向使用者的兜底
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def render_page(modes: list[str]) -> str:
    options = "\n".join(
        f'        <option value="{m}"{" selected" if m == "bm25" else ""}>'
        f"{MODE_LABELS.get(m, m)}</option>"
        for m in modes
    )
    return PAGE.replace("__MODE_OPTIONS__", options)


def make_server(
    rag: InfusionPumpRAG,
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    quiet: bool = False,
) -> ThreadingHTTPServer:
    """建好但还没启动的 HTTP 服务。

    单独暴露出来，一是让测试能拿 port=0 起随机端口，二是测试里可以关掉访问日志。
    """
    modes = rag.stats()["modes"]
    handler = _make_handler(rag, render_page(modes).encode("utf-8"), quiet=quiet)
    return ThreadingHTTPServer((host, port), handler)


def serve(
    index_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    rag = InfusionPumpRAG.load(index_path)
    stats = rag.stats()
    llm = LLMConfig.from_env()
    modes = stats["modes"]

    httpd = make_server(rag, host, port)

    print("\n输液泵 / 注射泵 知识库")
    print(f"  文档 {stats['n_documents']} 篇 · 片段 {stats['n_chunks']} 个 · 词表 {stats['vocab_size']}")
    print(f"  检索模式：{', '.join(modes)}")
    print(f"  生成方式：{'大模型 ' + llm.model if llm.enabled else '抽取式（未配置大模型）'}")
    print(f"\n  浏览器打开 http://{host}:{port}\n")
    print("  Ctrl+C 停止\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
