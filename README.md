# 书童 (Page)：基于《人生进阶指南》的本地学习伴侣

> 书童的职守有三：**背书**（RAG 检索——背的就是这本《人生进阶指南》）、
> **记日程**（每日任务与 90 天计划）、**管笔墨**（证据存档与周复盘）。
> 英文名 Page，一词双关：书页，与随侍的小书童。

第一阶段骨架：**Hybrid RAG 问答**（向量 + BM25 混合检索 + 查询改写 + 上下文增强），
语料是 [byoungd/up](https://github.com/byoungd/up)（人生进阶指南，含英语学习篇与 AI 学习篇）。

> 注意：该书内容为 CC BY-NC 4.0（署名-非商业）。自用、练手没问题，别拿去做商业产品。

## 结构

```
rag-1/
├── config.py      # 路径与 LLM 参数（读 .env）
├── llm.py         # OpenAI 兼容接口封装：chat / embed / 改写 / 上下文生成
├── store.py       # SQLite 本地向量库（chunk 正文 + 上下文 + float32 向量）
├── retrieve.py    # 检索：向量余弦 + BM25（中文 bigram），RRF 融合
├── ingest.py      # 建库：切分 -> （可选）上下文增强 -> 向量化 -> 入库
├── ask.py         # 命令行问答（多轮、可选追问改写、带出处标注）
├── eval.py        # 检索评测：16 题，pass@k + MRR
├── eval.md        # 评测基线记录与验收标准
├── data/up/       # clone 下来的语料仓库
└── index/         # 建好的索引（chunks.db）
```

## 快速开始

```bat
:: 1. 装依赖（已有 data/up 语料则跳过 clone）
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

:: 2. 配 key：复制 .env.example 为 .env，填入 LLM_API_KEY
copy .env.example .env

:: 3. 看切分效果（不花钱）
.venv\Scripts\python ingest.py --dry-run

:: 3.5 没配 key 前可先只切分入库，体验 BM25 检索
.venv\Scripts\python ingest.py --chunk-only
.venv\Scripts\python ask.py --no-llm --mode bm25 证据卡

:: 4. 建库（调 embedding API）；加 --contextual 更准（有缓存，多花一次对话模型的钱）
.venv\Scripts\python ingest.py
.venv\Scripts\python ingest.py --contextual

:: 5. 提问与留痕
.venv\Scripts\python ask.py 口语很差怎么练     :: 每次问答自动留痕（问题/命中/答案入库）
.venv\Scripts\python ask.py --log 5           :: 回看最近 5 条留痕
.venv\Scripts\python ask.py --rate 3 bad --note "命中对但答案泛"   :: 打标；bad 自动回流进评测题库
.venv\Scripts\python eval.py                  :: 主榜 + 真实坏例区一起看（真实坏例补 expect 后计分）
.venv\Scripts\python ask.py --rewrite         :: 交互模式，支持多轮追问
.venv\Scripts\python ask.py --no-llm 证据卡   :: 只看检索命中，不调用任何模型
.venv\Scripts\python ask.py --mode bm25 证据卡 :: 对比 vector / bm25 / hybrid

:: 6. 评测（改动任何参数后跑一遍，分数不应下降）
.venv\Scripts\python eval.py
```

## 检索说明

- **vector**：embedding 余弦，语义相似但术语对不上时靠它
- **bm25**：中文 bigram + 英文分词的关键词检索，术语精确命中（如「证据卡」「CEFR」）
- **hybrid**：两路各取 top-k×3，RRF 融合，默认模式
- **--contextual**（建库时）：入库前让对话模型按文档大纲给每个 chunk 写一句
  上下文说明再向量化（Anthropic Contextual Retrieval 的轻量版），缓解
  "chunk 脱离上下文看不出相关性"的问题；结果按 chunk 哈希缓存，重建不重复花钱
- **--rewrite**（问答时）：多轮对话的追问先用对话模型改写成独立完整查询再检索，
  解决"它是什么""那听力呢"这类指代问题；无 key 时自动用原查询

## 评测基线（2026-10-28，仅切分、BM25 top-3）

pass@3 = 10/16，pass@1 = 3/16，MRR = 0.354。
6 个未通过全部是语义改写类问题，是接模型后 hybrid 的验收对象，**验收线 ≥14/16**。
详见 eval.md。

## 完整用法（三阶段全开）

```bat
:: 第一阶段：问答与评测
.venv\Scripts\python ask.py --rewrite                :: 日常问答（多轮、追问改写）
.venv\Scripts\python eval.py                         :: 检索回归评测

:: 第二阶段：诊断与计划
.venv\Scripts\python diagnose.py                     :: 10 分钟自测 → 存档诊断
.venv\Scripts\python plan.py                         :: 生成 90 天计划 → 存档
.venv\Scripts\python plan.py --show                  :: 回看计划

:: 第三阶段：每日陪跑
.venv\Scripts\python daily.py                        :: 派发今天的 25-45 分钟任务
.venv\Scripts\python daily.py --done --note "听了XX，抓到X个词块，产出在XX"   :: 打卡+证据存档
.venv\Scripts\python daily.py --skip                 :: 今天跳过（留痕）
.venv\Scripts\python review.py                       :: 周日复盘：只依据证据，输出下周唯一变量
.venv\Scripts\python review.py --show                :: 回看复盘
```

陪跑闭环：`daily.py` 派发任务（参考计划+诊断+证据+最近复盘）→ 你交证据 →
`review.py` 只依据证据复盘并给出「下周只改变的一个变量」→ 影响下一天的任务派发。

## 许可

代码与文档：MIT（见 LICENSE）。检索语料《人生进阶指南》：CC BY-NC 4.0（不在本仓库内，仅方法可迁移）。

## 模拟测试

`answers.sample*.json` 是 9 个测试画像（程序员 B1 / 零基础家长 / 酒后自信 / 考研学生 /
敷衍用户 / 10岁儿童 / 68岁老人 / 海归研究员 C1 / 学中文的外国人），
覆盖不同水平、年龄、身份和输入质量。
`simulate_week.py` 模拟一周真实使用节奏（好证据/模糊证据/跳过混合），检验闭环状态：

```bat
.venv\Scripts\python simulate_week.py answers.sample4.json --fresh
:: --fresh 清空任务/证据/复盘（模拟新用户），诊断与计划历史保留
```

## 模拟测试发现并修复过的坑（设计备忘）

- **跨状态污染**：换画像后旧复盘盖过新计划 → daily 派发加了权威优先级（计划>诊断>复盘>证据）
- **任务复读**：小模型无视"不要重复"指令 → 机械去重（difflib 相似度 + 带批评重试）
- **自述谄媚**：盲目自信的输入骗到高分 → 诊断拆两段（客观取证 → 只依据证据定级）
- **模板模仿**：检索把书中填空模板带进上下文，模型照抄 → 治理检索源头而非提示词

## 更新语料

书仓库有更新时，缓存按内容哈希生效，只有变动的 chunk 会重新花钱：

```bat
cd data\up && git pull && cd ..\..
.venv\Scripts\python ingest.py --contextual
.venv\Scripts\python eval.py    :: 分数不应下降；书结构大改时先核对 eval.py 的期望关键词
```

## 健壮性设计

- 建库"全部准备好才写库"，任何一步失败不影响现有索引
- API 重试：SDK 内置 5 次指数退避（429/5xx/网络抖动）；仍失败则报错退出，
  已完成的批次全部落缓存，重跑自动续
- 上下文和向量均按内容哈希缓存，失败结果不入缓存，重跑自动补齐
- 所有模型功能在无 key 时可关（--chunk-only / --no-llm / 自动降级）

## 之后的路线

- **第二阶段**：CEFR 自测诊断流 → 生成 90 天学习计划（还是 RAG，加结构化流程）
- **第三阶段**：每日任务派发 + 证据存档（SQLite 加表）+ 每周复盘（开始 agent 化）

当前所有检索/存储接口（`Retriever.search`、`store.*`）都按可替换设计，
后面换 Chroma/加 agent 不用推翻重来。
