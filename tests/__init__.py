"""tests 包标记，让 `from tests.fixtures import ...` 可用。

顺便把标准输出切成 UTF-8 —— Windows 默认的 GBK 会把中文断言消息打成乱码，
排查失败用例时很难受。
"""
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover
        pass
