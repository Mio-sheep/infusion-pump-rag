#!/usr/bin/env python
"""比较两份索引在忽略构建时间之后是否完全一致。

CI 里用它验证「仓库里提交的 index/kb_index.json」和「用当前 data/raw 现场重建的索引」
是不是同一份内容。不一致就说明有人改了语料忘了重新 build 并提交索引。

    python scripts/check_index_sync.py index/kb_index.json /tmp/fresh.json

为什么要忽略 built_at：那是墙钟时间，两次构建必然不同，
它不代表语料内容变了。corpus_dir 已经改成只存目录名，不再存本机绝对路径。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

IGNORED_META_KEYS = ("built_at",)


def normalized(path: str | Path) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    for key in IGNORED_META_KEYS:
        meta.pop(key, None)
    return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=1)


def first_difference(a: str, b: str) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            context = a[max(0, i - 70) : i + 70]
            return f"首个差异在第 {i} 个字符附近：…{context}…"
    return f"内容长度不同：{len(a)} vs {len(b)}"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("用法：check_index_sync.py <已提交的索引> <现场重建的索引>")
        return 2

    committed = normalized(argv[0])
    fresh = normalized(argv[1])

    if committed == fresh:
        print("索引与语料一致")
        return 0

    print("索引与语料不一致。请重新运行 build，并把 index/kb_index.json 一起提交：")
    print(f"  {first_difference(committed, fresh)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
