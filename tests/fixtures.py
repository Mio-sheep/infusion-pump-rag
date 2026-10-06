"""测试用的迷你语料与临时索引。

刻意做成"小而完整"：两篇文档，带有前置元数据、表格、超长段落和一个
"参考来源"小节，覆盖分块器里所有分支。
"""
from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OCCLUSION_DOC = """---
title: 阻塞报警处置
summary: 上游阻塞与下游阻塞的区别、解除阻塞时的团注风险
updated: 2026-02-28
sources:
  - label: FDA, Examples of Reported Infusion Pump Problems
    url: https://example.org/fda-problems
  - label: IEC 60601-2-24
    url: https://example.org/iec
---

# 阻塞报警处置

## 上游阻塞

上游阻塞指泵与药液容器之间受阻，常见原因是夹子未打开、管路打折或穿刺器未插到底。
表现为泵持续报警但患者端仍有滴速；若不处理，最终会转为输注中断。

## 下游阻塞

下游阻塞指泵与患者之间受阻，常见原因是留置针贴壁、三通关闭或管路受压。
下游阻塞解除的瞬间可能出现一次性团注，对小容量高浓度药液风险显著。

## 压力档位对照表

| 档位 | 常见阈值范围 | 适用场景 | 备注 |
| --- | --- | --- | --- |
| 低 | 约 100 mmHg 以下 | 中心静脉、小容量 | 报警早，误报多 |
| 中 | 约 100 至 300 mmHg | 常规外周静脉 | 最常用 |
| 高 | 约 300 mmHg 以上 | 高压管路、快速补液 | 报警晚，渗漏风险高 |

## 参考来源

- [FDA, Examples of Reported Infusion Pump Problems](https://example.org/fda-problems)
- [IEC 60601-2-24](https://example.org/iec)
"""

BATTERY_DOC = """---
title: 电池与电源
summary: 电池老化、过充与转运途中的断电风险
updated: 2026-02-28
sources:
  - label: FDA, Infusion Pump Risk Reduction Strategies for Clinicians
    url: https://example.org/fda-clinicians
---

# 电池与电源

## 失效表现

电池未按推荐寿命周期更换时，容量会缓慢下降，表现为续航时间明显短于标称值。
密封铅酸电池过充后会鼓包，外壳沿接缝开裂，继续使用存在过热风险。

## 转运注意

转运前应确认电量充足。低电量报警的扬声器音量如果被调低，报警可能完全被忽略，
电池耗尽后泵会自动关机。
"""


@contextmanager
def temp_project():
    """建一个临时语料目录 + 索引路径，退出时清理。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        corpus = root / "raw"
        corpus.mkdir()
        (corpus / "01-阻塞报警处置.md").write_text(OCCLUSION_DOC, encoding="utf-8")
        (corpus / "02-电池与电源.md").write_text(BATTERY_DOC, encoding="utf-8")
        yield corpus, root / "kb_index.json"


@contextmanager
def temp_rag(**build_kwargs):
    """建好索引并加载成 InfusionPumpRAG。"""
    from infusion_rag import InfusionPumpRAG

    with temp_project() as (corpus, index_path):
        rag = InfusionPumpRAG.build(
            corpus_dir=corpus, index_path=index_path, **build_kwargs
        )
        yield rag, corpus, index_path
