# 参与贡献

欢迎提 issue 和 PR。这个项目有两类改动，要求不太一样，分开说。

## 环境

不需要装任何东西。Python 3.9+ 自带的标准库就够了。

```bash
git clone https://github.com/Mio-sheep/infusion-pump-rag
cd infusion-pump-rag
python -m unittest discover -s tests -t . -v
```

只有在做语义检索相关改动时才需要可选项：

```bash
pip install -r requirements-optional.txt   # sentence-transformers
pip install ruff mypy                      # 静态检查
```

**请不要为了图方便给核心链路加第三方依赖。** "clone 下来零安装就能跑"是这个项目
最主要的取舍之一，CI 里专门有一条注释在守着它。

## 改代码

1. 跑一遍完整测试：`python -m unittest discover -s tests -t .`
2. 跑静态检查：`ruff check . && ruff format --check . && mypy`
3. **如果动了检索逻辑，必须附上评测对比**：

   ```bash
   python -m infusion_rag.cli eval --failures 10    # 改动前
   # ...改代码...
   python -m infusion_rag.cli eval --failures 10    # 改动后
   ```

   这个项目的立场是：检索质量的改动如果没有评测数字支撑，就不算改好了。
   凭"感觉更准了"调权重，最后一定会把别的地方调坏。

## 改知识库内容

内容在 `data/raw/*.md`。改完之后必须重新构建索引并一起提交：

```bash
python -m infusion_rag.cli build
git add data/raw index/kb_index.json
```

CI 会检查 `index/kb_index.json` 是否和 `data/raw` 同步，不同步会直接失败。

### 文件格式

每篇文档以一段前置元数据开头：

```markdown
---
title: 报警与故障排查
summary: 一句话说明这篇讲什么，会显示在文档列表里
updated: 2026-02-28
sources:
  - label: FDA, Examples of Reported Infusion Pump Problems
    url: https://www.fda.gov/medical-devices/infusion-pumps/examples-reported-infusion-pump-problems
  - label: 某型号输液泵使用说明书（2023 版）
    url: ""
---

# 正文标题

……
```

要点：

- `sources` 按**正文中引用它们的顺序**排列，正文里用 `[1]`、`[2]` 对应。
- 没有 URL 的出处（纸质手册、口述经验）也照写，`url` 留空即可。
- **不要把"参考来源"写成正文的最后一个小节**。出处是元数据，写进正文会被索引，
  导致关键词在网址里虚高命中。分块器确实会自动把这类小节剥离出来，但不如一开始就写对。
- `updated` 是你实际核对过内容准确性的日期，不是随手填的。

### 写作风格

这部分比格式重要。这个知识库是给人查资料用的，不是给搜索引擎凑关键词的。

**要写具体的东西。**

- 差：`阻塞报警可能由多种原因引起，需要根据实际情况排查。`
- 好：`下游阻塞最常见的原因是留置针贴壁。判断方法是把留置针翻转 90 度后报警是否消失。`

**说清楚每条结论的依据等级。** 有三种情况，写法不同：

| 依据等级 | 写法 |
| --- | --- |
| 标准/法规明确要求 | 直接陈述，并给出标准编号和条款号 |
| 厂家手册或权威文献 | 陈述，并标注型号/版本或文献出处 |
| 工程惯例或经验判断 | **必须明说**，例如"以下是实际排查中的经验排序，标准里没有规定" |

把第三种伪装成第一种，是这个领域最容易出的事，也是这个仓库最不希望出现的事。

**不要用模板腔。**

- 不要每段都用加粗小标题起头（`**要点：**` 这种）。
- 不要把一个自然段拆成五条 bullet —— 如果它们本来就是一句话能说清的关系，就写成一句话。
- 不要用"首先/其次/最后""综上所述""值得注意的是""不仅……而且"。
- 不要用 emoji。医疗设备文档里出现 emoji 会削弱可信度。
- 表格只在**结构上确实是表格**的时候用（参数对照、报警名称对照、标准编号对照）。
  把叙述性内容塞进两列表格是最典型的 AI 痕迹。
- 中英文混排时，英文术语和数字两侧留空格，但中文标点前后不留。

**不要编造。** 拿不准的型号、参数、条款号、页码，要么去查证，要么写"常见机型"
这样的模糊表述，要么不写。宁可少写一条，也不要写一条错的。

## 提交信息

用中文或英文都行，格式大致是 `<类型>: <做了什么>`，类型用
`feat` / `fix` / `docs` / `data` / `refactor` / `test` / `chore`。

```
data: 补充下游阻塞的判断方法
fix: BM25 标题字段在空标题时长度归一化除零
docs: README 补上评测结果表
```

## 关于内容的免责

这个仓库整理的是公开资料，不构成临床依据。提交内容时请确认你引用的材料
是可以公开引用的，不要把内部培训材料、受版权保护的付费标准原文贴进来。
引用标准时写编号和条款位置即可，不要整段复制标准正文。
