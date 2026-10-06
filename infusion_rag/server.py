"""网页版检索界面：只用标准库 http.server，无需 Flask / FastAPI。

    python -m infusion_rag.cli serve --port 8000
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import LLMConfig
from .pipeline import InfusionPumpRAG

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>输液泵 / 注射泵 知识库</title>
<style>
  :root { --bg:#0f1115; --card:#171a21; --line:#272b35; --fg:#e6e9ef; --muted:#8b93a7; --accent:#4f8cff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family: -apple-system, "Segoe UI", "Microsoft YaHei", Roboto, sans-serif; }
  header { padding:28px 20px 12px; max-width:900px; margin:0 auto; }
  h1 { font-size:22px; margin:0 0 6px; }
  .sub { color:var(--muted); font-size:13px; }
  .searchbar { display:flex; gap:10px; max-width:900px; margin:18px auto; padding:0 20px; }
  input[type=text] { flex:1; padding:13px 15px; border-radius:10px; border:1px solid var(--line);
                     background:var(--card); color:var(--fg); font-size:15px; outline:none; }
  input[type=text]:focus { border-color:var(--accent); }
  button { padding:13px 22px; border-radius:10px; border:0; background:var(--accent);
           color:#fff; font-size:15px; cursor:pointer; font-weight:600; }
  button:hover { filter:brightness(1.1); }
  button.ghost { background:var(--card); color:var(--fg); border:1px solid var(--line); font-weight:400; }
  .examples { max-width:900px; margin:0 auto 8px; padding:0 20px; display:flex; flex-wrap:wrap; gap:8px; }
  .chip { font-size:12.5px; color:var(--muted); border:1px solid var(--line); border-radius:999px;
          padding:5px 12px; cursor:pointer; background:transparent; }
  .chip:hover { color:var(--fg); border-color:var(--accent); }
  main { max-width:900px; margin:0 auto; padding:6px 20px 60px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:16px 18px; margin-bottom:14px; }
  .card h3 { margin:0 0 8px; font-size:15px; }
  .meta { color:var(--muted); font-size:12px; margin-bottom:10px; font-family: ui-monospace, Consolas, monospace; }
  pre { white-space:pre-wrap; word-break:break-word; margin:0; font-size:13.5px; line-height:1.65;
        font-family: inherit; }
  table { border-collapse:collapse; width:100%; font-size:13px; margin:8px 0; }
  th, td { border:1px solid var(--line); padding:6px 9px; text-align:left; }
  th { background:#1e222b; }
  .rank { display:inline-block; min-width:22px; height:22px; line-height:22px; text-align:center;
          border-radius:6px; background:var(--accent); color:#fff; font-size:12px;
          font-weight:700; margin-right:8px; }
  .answer { border-color:#2b4a86; background:#141a26; }
  .answer h3 { color:var(--accent); }
  .status { color:var(--muted); font-size:13px; padding:6px 0 18px; }
  .switches { max-width:900px; margin:0 auto; padding:0 20px 6px; color:var(--muted); font-size:12.5px;
              display:flex; gap:16px; align-items:center; }
  select { background:var(--card); color:var(--fg); border:1px solid var(--line);
           border-radius:8px; padding:5px 8px; }
  footer { max-width:900px; margin:0 auto; padding:0 20px 40px; color:var(--muted); font-size:12px;
           line-height:1.7; border-top:1px solid var(--line); padding-top:16px; }
</style>
</head>
<body>
<header>
  <h1>输液泵 / 注射泵 知识库</h1>
  <div class="sub" id="stats">加载中…</div>
</header>

<div class="searchbar">
  <input type="text" id="q" placeholder="例如：阻塞报警怎么排查？JJF 1259 是什么？" autocomplete="off">
  <button id="go">检索</button>
  <button class="ghost" id="askBtn">问答</button>
</div>

<div class="switches">
  <label>检索模式
    <select id="mode">
      <option value="hybrid">hybrid（混合）</option>
      <option value="lexical">lexical（关键词）</option>
      <option value="dense">dense（语义）</option>
    </select>
  </label>
  <label>返回条数 <select id="k">
    <option>3</option><option selected>4</option><option>6</option><option>8</option>
  </select></label>
</div>

<div class="examples" id="examples"></div>
<main id="out"></main>

<footer>
  本知识库内容整理自 FDA 输液泵公开文档、JJF 1259-2010 / IEC 60601-2-24 等公开资料，
  仅供学习与检索参考，<b>不能替代设备说明书、院内操作规程与临床判断</b>。
</footer>

<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/[&<>]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function renderMarkdownLite(text) {
  let html = esc(text);
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<b>$1</b>');
  return html;
}

function renderHits(hits) {
  if (!hits.length) return '<div class="card">没有命中任何片段，换个说法试试。</div>';
  return hits.map((h) => `
    <div class="card">
      <h3><span class="rank">${h.rank}</span>${esc(h.doc_title)} › ${esc(h.heading || '（无小节）')}</h3>
      <div class="meta">${esc(h.source)} · ${esc(h.chunk_id)} · score=${h.score.toFixed(4)} (lex=${h.lexical.toFixed(4)}, dense=${h.dense.toFixed(4)})</div>
      <pre>${renderMarkdownLite(h.text)}</pre>
    </div>`).join('');
}

async function doSearch() {
  const q = $('q').value.trim();
  if (!q) return;
  const mode = $('mode').value, k = $('k').value;
  $('out').innerHTML = '<div class="status">检索中…</div>';
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&k=${k}&mode=${mode}`);
    const data = await res.json();
    if (data.error) { $('out').innerHTML = `<div class="card">错误：${esc(data.error)}</div>`; return; }
    $('out').innerHTML = `<div class="status">命中 ${data.n_hits} 条 · 模式 ${data.mode}</div>` + renderHits(data.hits);
  } catch (e) {
    $('out').innerHTML = `<div class="card">请求失败：${esc(String(e))}</div>`;
  }
}

async function doAsk() {
  const q = $('q').value.trim();
  if (!q) return;
  const k = $('k').value;
  $('out').innerHTML = '<div class="status">生成中…</div>';
  try {
    const res = await fetch(`/api/ask?q=${encodeURIComponent(q)}&k=${k}`);
    const data = await res.json();
    if (data.error) { $('out').innerHTML = `<div class="card">错误：${esc(data.error)}</div>`; return; }
    const answer = `<div class="card answer"><h3>答案 · ${data.generator}</h3><pre>${renderMarkdownLite(data.answer)}</pre></div>`;
    $('out').innerHTML = answer +
      `<div class="status">以下为支撑答案的原文片段</div>` + renderHits(data.hits);
  } catch (e) {
    $('out').innerHTML = `<div class="card">请求失败：${esc(String(e))}</div>`;
  }
}

const EXAMPLES = [
  '阻塞报警怎么排查',
  '气泡报警可能是什么原因',
  '独立双人核对的含义',
  'JJF 1259 是什么标准',
  '电池为什么会导致泵停机',
  '过量输注 over-infusion 的定义',
];

$('examples').innerHTML = EXAMPLES.map((e) => `<button class="chip">${e}</button>`).join('');
$('examples').addEventListener('click', (ev) => {
  if (ev.target.classList.contains('chip')) { $('q').value = ev.target.textContent; doSearch(); }
});
$('go').addEventListener('click', doSearch);
$('askBtn').addEventListener('click', doAsk);
$('q').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') doSearch(); });

fetch('/api/stats').then((r) => r.json()).then((s) => {
  $('stats').textContent = `${s.n_documents} 篇文档 · ${s.n_chunks} 个片段 · 词表 ${s.vocab_size} · 平均 ${s.avg_chunk_chars} 字/片段`
    + (s.has_dense ? ` · 语义向量：${s.dense_model}` : ' · 仅关键词检索（tfidf）');
}).catch(() => { $('stats').textContent = '统计信息不可用'; });
</script>
</body>
</html>
"""


def _make_handler(rag: InfusionPumpRAG):
    class Handler(BaseHTTPRequestHandler):
        server_version = "infusion-pump-rag/1.0"

        # -- 工具方法 --------------------------------------------------- #
        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8", status)

        def log_message(self, fmt, *args):  # noqa: D102
            print(f"  {self.address_string()} {fmt % args}")

        # -- 路由 -------------------------------------------------------- #
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            path = parsed.path.rstrip("/") or "/"

            if path in {"/", "/index.html"}:
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/stats":
                self._json(rag.stats())
                return

            if path == "/api/docs":
                self._json({"documents": rag.documents()})
                return

            if path in {"/api/search", "/api/ask"}:
                text = (query.get("q") or [""])[0].strip()
                if not text:
                    self._json({"error": "缺少参数 q"}, 400)
                    return
                try:
                    top_k = max(1, min(int((query.get("k") or ["4"])[0]), 20))
                except ValueError:
                    top_k = 4
                mode = (query.get("mode") or ["hybrid"])[0]
                try:
                    if path == "/api/search":
                        hits = rag.search(text, top_k=top_k, mode=mode)
                        self._json(
                            {
                                "query": text,
                                "mode": mode,
                                "n_hits": len(hits),
                                "hits": [h.to_dict() for h in hits],
                            }
                        )
                    else:
                        use_llm = (query.get("llm") or ["auto"])[0]
                        result = rag.ask(
                            text,
                            top_k=top_k,
                            mode=mode,
                            use_llm=False if use_llm == "0" else (True if use_llm == "1" else None),
                        )
                        self._json(result)
                except Exception as exc:  # pragma: no cover - 面向用户的兜底
                    self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
                return

            self._json({"error": "not found"}, 404)

    return Handler


def serve(index_path: str | Path | None = None, host: str = "127.0.0.1",
          port: int = 8000) -> None:
    rag = InfusionPumpRAG.load(index_path)
    stats = rag.stats()
    httpd = ThreadingHTTPServer((host, port), _make_handler(rag))
    llm = LLMConfig.from_env()

    print(f"\n输液泵 / 注射泵 知识库")
    print(f"  文档 {stats['n_documents']} 篇 · 片段 {stats['n_chunks']} 个 · 词表 {stats['vocab_size']}")
    print(f"  生成方式：{'大模型 ' + llm.model if llm.enabled else '抽取式（未配置大模型）'}")
    print(f"\n  打开浏览器访问： http://{host}:{port}\n")
    print("  按 Ctrl+C 停止\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
