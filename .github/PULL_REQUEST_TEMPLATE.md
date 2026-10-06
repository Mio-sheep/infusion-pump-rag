## 这个 PR 做了什么

<!-- 一两句话说明。如果是改知识库内容，请写清楚依据。 -->

## 类型

- [ ] 修 bug
- [ ] 改检索 / 索引 / 评测
- [ ] 改知识库内容
- [ ] 改文档或工程配置

## 检查清单

- [ ] `python -m unittest discover -s tests -t .` 全过
- [ ] 改了 `data/raw/` 的话，重新跑了 `build` 并提交了 `index/kb_index.json`
- [ ] 改了检索逻辑的话，附上了 `python -m infusion_rag.cli eval` 前后的对比
- [ ] 新增知识库内容的话，出处写进了文件的前置元数据

## 评测对比（改了检索逻辑时填）

| | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| 改动前 | | | | |
| 改动后 | | | | |
