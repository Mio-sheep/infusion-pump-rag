"""命令行入口。

python -m infusion_rag.cli build                 构建索引
python -m infusion_rag.cli ask "阻塞报警怎么处理"
python -m infusion_rag.cli search "JJF 1259" -k 3
python -m infusion_rag.cli explain "阻塞报警"     看检索细节（分词、两路分数）
python -m infusion_rag.cli eval                  跑评测集
python -m infusion_rag.cli docs                  列出语料文档与出处
python -m infusion_rag.cli stats                 索引统计
python -m infusion_rag.cli serve --port 8000     网页版
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from .config import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_EVAL_PATH,
    DEFAULT_INDEX_PATH,
    BM25Config,
    ChunkConfig,
)
from .dense import DEFAULT_MODEL, INSTALL_HINT
from .pipeline import InfusionPumpRAG
from .retriever import MODES

PROG = "infusion-rag"


def _force_utf8() -> None:
    """Windows 控制台默认可能是 GBK，中文输出会炸。"""
    for stream in (sys.stdout, sys.stderr):
        # 只有真正的终端对象才有 reconfigure；重定向到文件时没有，静默跳过即可
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def _print_hits(hits, *, show_text: bool = True, width: int = 400) -> None:
    if not hits:
        print("  （没有命中）")
        return
    for hit in hits:
        print(f"  [{hit.rank}] {hit.label}")
        print(
            f"      {hit.chunk.path} · {hit.chunk.chunk_id} · "
            f"score={hit.score:.4f} (bm25={hit.bm25:.3f}, dense={hit.dense:.3f})"
        )
        if show_text:
            text = " ".join(hit.chunk.text.split())
            print(f"      {text[:width]}{'…' if len(text) > width else ''}")
        print()


def _add_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query")
    parser.add_argument("-k", "--top-k", type=int, default=4, help="返回条数")
    parser.add_argument("-m", "--mode", default="bm25", choices=MODES, help="检索模式（默认 bm25）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="输液泵 / 注射泵 小型 RAG 知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--index", default=str(DEFAULT_INDEX_PATH), help="索引文件（默认 index/kb_index.json）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- build ---------------------------------------------------------- #
    p = sub.add_parser("build", help="从 data/raw 构建索引")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS_DIR), help="语料目录")
    p.add_argument("--dense", action="store_true", help="同时生成稠密向量（需要可选依赖）")
    p.add_argument("--dense-model", default=DEFAULT_MODEL, help="稠密向量模型")
    p.add_argument("--max-chars", type=int, default=ChunkConfig.max_chars)
    p.add_argument("--overlap", type=int, default=ChunkConfig.overlap_chars)
    p.add_argument(
        "--heading-weight",
        type=float,
        default=BM25Config.heading_weight,
        help="标题字段权重（默认 2.0）",
    )
    p.add_argument("--k1", type=float, default=BM25Config.k1)
    p.add_argument("--b", type=float, default=BM25Config.b)

    # -- search / ask / explain ----------------------------------------- #
    _add_query_args(sub.add_parser("search", help="只检索，不生成"))
    sub.choices["search"].add_argument("--no-text", action="store_true", help="不打印片段正文")

    p = sub.add_parser("ask", help="检索 + 生成答案")
    _add_query_args(p)
    p.add_argument("--no-llm", action="store_true", help="强制用抽取式回答，不调大模型")

    p = sub.add_parser("explain", help="查看检索细节，用于调试")
    p.add_argument("query")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("-m", "--mode", default="bm25", choices=MODES)

    # -- eval ------------------------------------------------------------ #
    p = sub.add_parser("eval", help="在评测集上算 Recall@k 与 MRR")
    p.add_argument("--questions", default=str(DEFAULT_EVAL_PATH), help="评测集文件")
    p.add_argument("-m", "--mode", default="bm25", choices=MODES)
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("--failures", type=int, default=0, help="最多列出多少条未命中的问题")
    p.add_argument("--json", action="store_true")

    # -- docs / stats / serve -------------------------------------------- #
    sub.add_parser("docs", help="列出语料文档与出处")
    sub.add_parser("stats", help="索引统计")

    p = sub.add_parser("calc", help="剂量率 ↔ 泵速 换算（不需要索引，随时可用）")
    p.add_argument("dose", nargs="?", help='剂量率，如 "0.1ug/kg/min" 或 "5mg/h"')
    p.add_argument("--conc", required=True, help='药液浓度，如 "4mg/50mL" 或 "80ug/mL"')
    p.add_argument("--weight", type=float, default=None, help="体重（kg），剂量率含 /kg 时必填")
    p.add_argument("--vtbi", type=float, default=None, help="待输容量（mL），用来估算输完时间")
    p.add_argument("--rate", default=None, help="反算：给泵上的 mL/h，算回剂量率")
    p.add_argument(
        "--as", dest="as_unit", default="ug/kg/min", help="反算的目标单位（默认 ug/kg/min）"
    )

    p = sub.add_parser("serve", help="启动网页版界面")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)

    # ---- build --------------------------------------------------------- #
    if args.command == "build":
        try:
            rag = InfusionPumpRAG.build(
                corpus_dir=args.corpus,
                index_path=args.index,
                chunk_cfg=ChunkConfig(max_chars=args.max_chars, overlap_chars=args.overlap),
                bm25_cfg=BM25Config(k1=args.k1, b=args.b, heading_weight=args.heading_weight),
                dense=args.dense,
                dense_model=args.dense_model,
                verbose=True,
            )
        except RuntimeError as exc:
            print(f"\n构建失败：{exc}", file=sys.stderr)
            if args.dense:
                print(INSTALL_HINT, file=sys.stderr)
            return 1

        stats = rag.stats()
        print(
            f"\n文档 {stats['n_documents']} 篇 · 片段 {stats['n_chunks']} 个 · "
            f"小节 {stats['n_sections']} 个 · 词表 {stats['vocab_size']} · "
            f"平均 {stats['avg_chunk_chars']} 字/片段 · "
            f"稠密向量 {'有' if stats['has_dense'] else '无'}"
        )
        return 0

    # ---- calc 不依赖索引，放在索引检查之前 -------------------------------- #
    if args.command == "calc":
        from .calc import CalcError, dose_to_rate, rate_to_dose

        try:
            if args.rate:
                outcome = rate_to_dose(args.rate, args.weight, args.conc, as_unit=args.as_unit)
            elif args.dose:
                outcome = dose_to_rate(args.dose, args.weight, args.conc, vtbi_ml=args.vtbi)
            else:
                print("要么给剂量率参数，要么用 --rate 指定泵速。详见 --help。", file=sys.stderr)
                return 2
        except CalcError as exc:
            print(f"输入有问题：{exc}", file=sys.stderr)
            return 2

        print()
        print(outcome.render())
        print()
        return 0

    # ---- 其余命令都需要已有索引 ------------------------------------------ #
    if not Path(args.index).exists():
        print(f"找不到索引 {args.index}\n请先运行：python -m {PROG} build", file=sys.stderr)
        return 2

    rag = InfusionPumpRAG.load(args.index)

    if args.command == "search":
        hits = rag.search(args.query, top_k=args.top_k, mode=args.mode)
        if args.json:
            print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
        else:
            print(f"\n查询：{args.query}    模式：{args.mode}\n")
            _print_hits(hits, show_text=not args.no_text)
        return 0

    if args.command == "ask":
        result = rag.ask(
            args.query,
            top_k=args.top_k,
            mode=args.mode,
            use_llm=False if args.no_llm else None,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("\n" + "=" * 72)
            print(result["answer"])
            print("=" * 72)
            print(
                f"\n生成方式：{result['generator']}    检索模式：{result['mode']}    "
                f"命中 {result['n_hits']} 条\n"
            )
            _print_hits(rag.search(args.query, top_k=args.top_k, mode=args.mode), show_text=False)
        return 0

    if args.command == "explain":
        print()
        print(rag.retriever.explain(args.query, mode=args.mode, top_k=args.top_k))
        print()
        return 0

    if args.command == "eval":
        try:
            report = rag.evaluate(args.questions, mode=args.mode, top_k=args.top_k)
        except FileNotFoundError as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(report.summary(), ensure_ascii=False, indent=2))
        else:
            print(f"\n评测集：{args.questions}")
            print(f"检索模式：{args.mode}\n")
            print(report.to_markdown(show_failures=args.failures))
            print()
        return 0

    if args.command == "docs":
        rows = rag.documents()
        print(f"\n共 {len(rows)} 篇文档\n")
        for row in rows:
            print(f"  {row['title']}")
            print(
                f"    {row['path']}  ·  {row['n_chunks']} 片段 / {row['n_chars']} 字"
                f"  ·  更新 {row['updated'] or '未标注'}"
            )
            if row["summary"]:
                print(f"    {row['summary']}")
            for source in row["sources"]:
                print(f"    ↳ {source['label']}")
            print()
        return 0

    if args.command == "stats":
        print()
        for key, value in rag.stats().items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value) if value else "—"
            print(f"  {key:<20} {value}")
        print()
        return 0

    if args.command == "serve":
        from .server import serve

        serve(index_path=args.index, host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
