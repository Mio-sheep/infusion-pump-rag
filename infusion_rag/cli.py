"""命令行入口。

    python -m infusion_rag.cli build                 构建索引
    python -m infusion_rag.cli ask "输液泵阻塞报警怎么处理"
    python -m infusion_rag.cli search "JJF 1259"
    python -m infusion_rag.cli docs                  列出语料文档
    python -m infusion_rag.cli stats                 查看索引统计
    python -m infusion_rag.cli serve --port 8000     启动网页版
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_CORPUS_DIR, DEFAULT_INDEX_PATH, ChunkConfig, EmbedderConfig
from .pipeline import InfusionPumpRAG

BANNER = "输液泵 / 注射泵 知识库 RAG · infusion-pump-rag"


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass


def _print_hits(hits, show_text: bool = True, width: int = 400) -> None:
    if not hits:
        print("  （没有命中）")
        return
    for hit in hits:
        print(f"  [{hit.rank}] {hit.chunk.doc_title} › {hit.chunk.heading}")
        print(f"      文件 {hit.chunk.source} · {hit.chunk.chunk_id} · "
              f"score={hit.score:.4f} (lex={hit.lexical:.4f}, dense={hit.dense:.4f})")
        if show_text:
            text = " ".join(hit.chunk.text.split())
            print(f"      {text[:width]}{'…' if len(text) > width else ''}")
        print()


def main(argv: list[str] | None = None) -> int:
    _force_utf8()

    parser = argparse.ArgumentParser(
        prog="infusion-rag",
        description=BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH),
                        help=f"索引文件路径（默认 {DEFAULT_INDEX_PATH}）")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- build ---------------------------------------------------------- #
    p_build = sub.add_parser("build", help="从 data/raw 构建索引")
    p_build.add_argument("--corpus", default=str(DEFAULT_CORPUS_DIR), help="语料目录")
    p_build.add_argument("--backend", default="tfidf",
                         choices=["tfidf", "sentence-transformers", "auto"],
                         help="向量后端（默认 tfidf，纯标准库）")
    p_build.add_argument("--model", default=None, help="sentence-transformers 模型名")
    p_build.add_argument("--max-chars", type=int, default=ChunkConfig.max_chars)
    p_build.add_argument("--overlap", type=int, default=ChunkConfig.overlap_chars)

    # -- search --------------------------------------------------------- #
    p_search = sub.add_parser("search", help="只检索，不生成")
    p_search.add_argument("query")
    p_search.add_argument("-k", "--top-k", type=int, default=4)
    p_search.add_argument("-m", "--mode", default="hybrid",
                          choices=["hybrid", "lexical", "dense"])
    p_search.add_argument("--json", action="store_true", help="输出 JSON")
    p_search.add_argument("--no-text", action="store_true", help="不打印片段正文")

    # -- ask ------------------------------------------------------------ #
    p_ask = sub.add_parser("ask", help="检索 + 生成答案")
    p_ask.add_argument("query")
    p_ask.add_argument("-k", "--top-k", type=int, default=4)
    p_ask.add_argument("-m", "--mode", default="hybrid",
                       choices=["hybrid", "lexical", "dense"])
    p_ask.add_argument("--no-llm", action="store_true", help="强制只用抽取式回答")
    p_ask.add_argument("--json", action="store_true")

    # -- docs / stats ---------------------------------------------------- #
    sub.add_parser("docs", help="列出语料文档")
    sub.add_parser("stats", help="查看索引统计")

    # -- serve ----------------------------------------------------------- #
    p_serve = sub.add_parser("serve", help="启动网页版检索界面")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    # ---- build ---------------------------------------------------------- #
    if args.command == "build":
        model = args.model
        embedder_cfg = EmbedderConfig(backend=args.backend)
        if model:
            embedder_cfg.model_name = model
        InfusionPumpRAG.build(
            corpus_dir=args.corpus,
            index_path=args.index,
            chunk_cfg=ChunkConfig(max_chars=args.max_chars, overlap_chars=args.overlap),
            embedder_cfg=embedder_cfg,
            verbose=True,
        )
        rag = InfusionPumpRAG.load(args.index)
        s = rag.stats()
        print(f"\n文档 {s['n_documents']} 篇 · 片段 {s['n_chunks']} 个 · "
              f"词表 {s['vocab_size']} · 平均片段 {s['avg_chunk_chars']} 字 · "
              f"稠密向量 {'有' if s['has_dense'] else '无'}")
        return 0

    # ---- 其余命令都需要已有索引 ------------------------------------------ #
    if not Path(args.index).exists():
        print(f"找不到索引文件 {args.index}\n请先运行：python -m infusion_rag.cli build",
              file=sys.stderr)
        return 2

    rag = InfusionPumpRAG.load(args.index)

    if args.command == "search":
        hits = rag.search(args.query, top_k=args.top_k, mode=args.mode)
        if args.json:
            print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        else:
            print(f"\n查询：{args.query}   模式：{args.mode}\n")
            _print_hits(hits, show_text=not args.no_text)
        return 0

    if args.command == "ask":
        result = rag.ask(args.query, top_k=args.top_k, mode=args.mode,
                         use_llm=False if args.no_llm else None)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"\n{'=' * 68}\n{result['answer']}\n{'=' * 68}")
            print(f"\n生成方式：{result['generator']}   检索模式：{result['mode']}"
                  f"   命中 {result['n_hits']} 条\n")
            _print_hits(rag.search(args.query, top_k=args.top_k, mode=args.mode),
                        show_text=False)
        return 0

    if args.command == "docs":
        rows = rag.documents()
        print(f"\n共 {len(rows)} 篇文档：\n")
        for row in rows:
            print(f"  · {row['title']}")
            print(f"      {row['source']}  ({row['chunks']} 片段 / {row['chars']} 字)")
        print()
        return 0

    if args.command == "stats":
        stats = rag.stats()
        print(f"\n{BANNER}\n")
        for key, value in stats.items():
            if isinstance(value, list):
                value = f"{len(value)} 项"
            print(f"  {key:<20} {value}")
        print()
        return 0

    if args.command == "serve":
        from .server import serve

        serve(index_path=args.index, host=args.host, port=args.port)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
