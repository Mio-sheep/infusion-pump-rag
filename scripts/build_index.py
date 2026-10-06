#!/usr/bin/env python
"""构建索引的便捷脚本（等价于 python -m infusion_rag.cli build）。

用法：
    python scripts/build_index.py
    python scripts/build_index.py --backend sentence-transformers
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infusion_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["build", *argv]
    raise SystemExit(main(argv))
