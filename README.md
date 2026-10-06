# 输液泵 / 注射泵 小型 RAG 知识库

一个**零依赖、可离线运行**的垂直领域 RAG 示例：知识库内容围绕 **输液泵（infusion pump）** 与 **注射泵（syringe pump）**，覆盖原理结构、临床使用规范、报警故障排查、风险与不良事件、维护校准、术语标准。

- **不装任何第三方包就能跑**：分块、TF-IDF 向量化、检索、抽取式问答、网页界面全部基于 Python 标准库实现。
- **不需要联网下载模型**：默认用中英文字符 n-gram TF-IDF 做检索，clone 下来即刻可用。
- **可选升级**：装了 `sentence-transformers` 就能开启语义向量并做混合检索；配置任意 OpenAI 兼容接口就能让大模型基于检索结果作答并标注出处。

> ⚠️ 本项目是**学习与检索演示**用途。知识库内容整理自公开资料，**不能替代设备说明书、院内操作规程与临床判断**。

---

## 目录

- [快速开始](#快速开始)
- [网页版界面](#网页版界面)
- [接入大模型](#接入大模型)
- [语义检索（可选）](#语义检索可选)
- [命令行参考](#命令行参考)
- [换成你自己的知识库](#换成你自己的知识库)
- [检索原理](#检索原理)
- [知识库内容](#知识库内容)
- [项目结构](#项目结构)
- [数据来源](#数据来源)
- [免责声明](#免责声明)
- [License](#license)

---

## 快速开始

只需要 **Python 3.9+**，没有其他依赖。

```bash
git clone <这个仓库的地址>
cd infusion-pump-rag

# 1) 构建索引（仓库里已经带了一份构建好的索引，这步可以跳过）
python -m infusion_rag.cli build

# 2) 提问
python -m infusion_rag.cli ask "阻塞报警应该怎么排查"

# 3) 只检索，不生成
python -m infusion_rag.cli search "JJF 1259 是什么标准" -k 3
```

`ask` 的输出示例：

```
====================================================================
**问题：** 阻塞报警应该怎么排查

未配置大模型，以下是知识库中与问题最相关的 4 个片段（按相关度排序）：

**[1] 输液泵与注射泵：常见报警与故障排查 › 2. 按报警类型排查 › 2.2 阻塞报警（Occlusion）**
`04-常见报警与故障排查.md` · 相关度 0.3580

### 2.2 阻塞报警（Occlusion）

可能原因：
- 管路打折、受压（患者压住、床栏挤压）。
- 三通/夹子处于关闭状态。
...
====================================================================
```

跑测试：

```bash
python -m unittest discover -s tests -v
```

---

## 网页版界面

只用标准库 `http.server`，不需要 Flask / FastAPI：

```bash
python -m infusion_rag.cli serve --port 8000
# 然后浏览器打开 http://127.0.0.1:8000
```

界面上可以实时切换 **hybrid / lexical / dense** 三种检索模式、调整返回条数，
`问答` 按钮会走"检索 + 生成"流程并在答案下方列出支撑它的原文片段。

JSON 接口：

| 接口 | 说明 |
| --- | --- |
| `GET /api/search?q=阻塞报警&k=4&mode=hybrid` | 返回排序后的片段 |
| `GET /api/ask?q=阻塞报警&k=4` | 返回答案 + 支撑片段（`llm=0` 强制抽取式） |
| `GET /api/stats` | 索引统计 |
| `GET /api/docs` | 语料文档清单 |

---

## 接入大模型

任意 **OpenAI 兼容** 的 `/chat/completions` 接口都可以。用环境变量配置：

```bash
# Linux / macOS
export RAG_LLM_BASE_URL="https://api.deepseek.com/v1"
export RAG_LLM_API_KEY="sk-xxxxxxxx"
export RAG_LLM_MODEL="deepseek-chat"

# Windows PowerShell
$env:RAG_LLM_BASE_URL="https://api.deepseek.com/v1"
$env:RAG_LLM_API_KEY="sk-xxxxxxxx"
$env:RAG_LLM_MODEL="deepseek-chat"

python -m infusion_rag.cli ask "气泡报警可能是什么原因"
```

配置好之后：

- `ask` 会自动改用大模型生成答案，并要求**逐条标注 `[编号]` 出处**；
- `serve` 的问答按钮同样会走大模型；
- 想让某次提问强制走抽取式（不调大模型），加 `--no-llm`。

> 没配大模型也完全能用：`ask` 会退化为"把最相关的原文片段排好队并标出处"，
> 对检索式使用来说信息量是一样的，而且**不会产生幻觉**。

---

## 语义检索（可选）

默认的 TF-IDF 是**关键词**检索：问"泵一直响个不停"不一定能命中"报警"。
想要语义检索就装可选项：

```bash
pip install -r requirements-optional.txt

# 重建索引，同时生成稠密向量
python -m infusion_rag.cli build --backend sentence-transformers

# 三种模式随便切
python -m infusion_rag.cli search "泵一直响个不停" --mode dense
python -m infusion_rag.cli search "泵一直响个不停" --mode hybrid
```

- `--backend tfidf`（默认）：只用稀疏向量，零依赖。
- `--backend sentence-transformers`：稀疏 + 稠密都建，`hybrid` 模式下用 **RRF（Reciprocal Rank Fusion）** 融合两路结果。
- `--backend auto`：装了就用语义模型，没装自动退回 tfidf。

中文语义模型默认用 `shibing624/text2vec-base-chinese`，可以用 `--model` 换：

```bash
python -m infusion_rag.cli build --backend sentence-transformers --model BAAI/bge-small-zh-v1.5
```

---

## 命令行参考

```bash
python -m infusion_rag.cli --help
```

| 命令 | 说明 |
| --- | --- |
| `build` | 读取 `data/raw/` 下的 Markdown，切块、向量化、写出索引 |
| `search "<查询>"` | 只检索。支持 `-k` 条数、`-m` 模式、`--json`、`--no-text` |
| `ask "<问题>"` | 检索 + 生成答案。支持 `--no-llm` 强制抽取式 |
| `docs` | 列出语料文档与各自片段数 |
| `stats` | 索引统计（文档数、片段数、词表、平均长度…） |
| `serve` | 启动网页界面，`--host` / `--port` |

`build` 的常用参数：

```bash
python -m infusion_rag.cli build \
  --corpus data/raw \
  --index index/kb_index.json \
  --backend tfidf \
  --max-chars 600 \
  --overlap 100
```

---

## 换成你自己的知识库

1. 把 Markdown / 纯文本文件丢进 `data/raw/`（会递归扫描 `.md` / `.markdown` / `.txt`）。
2. 重新构建：`python -m infusion_rag.cli build`
3. 完事。

分块规则（`infusion_rag/chunker.py`）：

- 先按 `#` 标题切成章节，小节标题会拼成 `一级 › 二级 › 三级` 的形式；
- 章节内先按空行分段，超长段落再按句号/分号切；
- Markdown 表格**按行切并给每块补上表头**，避免表格被切断后看不懂；
- 相邻块之间保留 `--overlap` 个字符的重叠，防止答案正好落在切口上。

如果你的语料是中文为主、术语很特殊，可以在 `infusion_rag/synonyms.py` 里补中英对照词，
中文查询会自动扩展出英文关键词（例如"阻塞" → `occlusion`），显著提升中英混排语料的召回。

---

## 检索原理

```
Markdown 语料
   │
   ├─ chunker.py    按标题/段落/表格切块（带重叠）
   │
   ├─ tokenizer.py  中英混排分词：CJK 用「单字 + 二元组」，拉丁文按词切
   │
   ├─ embedder.py   TF-IDF（L2 归一化，点积即余弦）  ← 默认，纯标准库
   │                + 可选 sentence-transformers 稠密向量
   │
   ├─ store.py      一个 JSON 装下 词表 + IDF + 稀疏向量 + 稠密向量
   │
   └─ retriever.py  两路打分 → RRF 融合 → 排序
         │
         ├─ 标题命中加成：查询词出现在小节标题里就加权
         ├─ 引用列表降权：全是网址的"参考来源"小节降低权重
         └─ 同义词扩展：中文查询自动补英文术语
   │
   └─ generator.py  抽取式（默认）或调用大模型生成并标注 [编号] 出处
```

为什么默认用**字符二元组**而不是分词器？因为中文专业词（"输液泵""蠕动泵""阻塞报警"）
用字符 bigram 就能召回，不需要维护词典、不需要 jieba，而且中英混排时不会把
`mL/h`、`IEC 60601-2-24` 这类单位/标准号切坏。

---

## 知识库内容

`data/raw/` 下共 **7 篇**文档（约 1.9 万字，切出 77 个片段）：

| 文档 | 内容 |
| --- | --- |
| `01-概述与分类.md` | 定义、按场景/用途/驱动原理分类、输液泵 vs 注射泵对比 |
| `02-工作原理与关键部件.md` | 蠕动泵/注射泵/弹性泵原理、关键部件失效点、按键连击、人因设计 |
| `03-临床使用与操作规范.md` | 预案、标识、核对、独立双人核对、五个正确、风险自查表 |
| `04-常见报警与故障排查.md` | 气泡/阻塞/注射器/门锁/电池/软件报警的排查思路与分级处置 |
| `05-风险与不良事件.md` | 过量/输注不足、四类根因、FDA 已报告问题、法规术语 |
| `06-维护保养与质量控制.md` | 清洁消毒、电池管理、PM 项目、JJF 1259 校准、不确定度 |
| `07-术语表与标准法规.md` | 中英术语对照、参数单位与换算、中外标准、不良事件上报 |

---

## 项目结构

```
infusion-pump-rag/
├── data/raw/                     # 知识库语料（Markdown）
├── index/kb_index.json           # 预构建索引（可直接用）
├── infusion_rag/
│   ├── config.py                 # 路径 / 分块 / 向量 / 大模型配置
│   ├── tokenizer.py              # 中英混排分词
│   ├── synonyms.py               # 查询扩展用的中英对照词表
│   ├── chunker.py                # 标题感知分块（含表格处理）
│   ├── embedder.py               # TF-IDF（标准库）+ sentence-transformers
│   ├── store.py                  # 索引序列化
│   ├── retriever.py              # 检索 + RRF 融合 + 加权
│   ├── generator.py              # 抽取式 / 大模型生成
│   ├── pipeline.py               # 串起整条流水线
│   ├── cli.py                    # 命令行
│   └── server.py                 # 网页界面（标准库 http.server）
├── scripts/build_index.py        # 构建索引的便捷脚本
├── tests/test_rag.py             # 单元测试（14 项）
├── pyproject.toml
├── requirements.txt              # 核心零依赖（说明文件）
└── requirements-optional.txt     # 可选的语义向量后端
```

---

## 数据来源

知识库内容整理、翻译并重述自以下公开资料（每篇文档末尾也标注了对应来源）：

- FDA · [What Is an Infusion Pump?](https://www.fda.gov/medical-devices/infusion-pumps/what-infusion-pump)
- FDA · [Infusion Pump: Glossary](https://www.fda.gov/medical-devices/infusion-pumps/infusion-pump-glossary)
- FDA · [Examples of Reported Infusion Pump Problems](https://www.fda.gov/medical-devices/infusion-pumps/examples-reported-infusion-pump-problems)
- FDA · [Infusion Pump Risk Reduction Strategies for Clinicians](https://www.fda.gov/medical-devices/infusion-pumps/infusion-pump-risk-reduction-strategies-clinicians)
- [JJF 1259-2010《医用注射泵和输液泵校准规范》](https://www.ndls.org.cn/standard/detail/b8b100eae987ffddcc3bddfd720d4a7e)（国家数字标准馆）
- [IEC 60601-2-24 标准更新说明](https://ewh.ieee.org/r6/ocs/pses/IEC%2060601-2%2024%20standard%20update%20requirements%20presentation.pdf)
- [对《医用注射泵和输液泵校准规范》(JJF 1259-2010) 的理解及建议](https://opaj.napstic.cn/periodicalArticle/0120170705974410)

---

## 免责声明

- 本项目为**软件工程与检索技术的学习示例**，不是医疗器械软件，也不是临床决策支持系统。
- 知识库内容源自公开资料的整理与重述，可能存在滞后、遗漏或理解偏差；**标准编号与版本请以现行有效文本为准**。
- 任何临床操作、设备维护与计量校准，请以**设备说明书、现行标准、所在机构规章制度**为准。
- 作者不对因使用本项目内容而产生的任何后果承担责任。

---

## License

代码以 [MIT License](LICENSE) 发布。知识库文档的来源与使用限制见 LICENSE 末尾说明。
