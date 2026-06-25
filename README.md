# CompIntel Research

> **Competitive Intelligence Agent System** — 输入一个公司名，10 节点 Agent Pipeline 协作完成竞品画像、市场分析、SWOT 综合、统稿审核与 RAG 自增长记忆。

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-ff69b4?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![RAG](https://img.shields.io/badge/RAG-Qdrant%20Hybrid-9cf?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![LLM](https://img.shields.io/badge/LLM-DeepSeek%20%7C%20Kimi%20%7C%20GLM-8A2BE2)](https://platform.deepseek.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%20%2B%20WebSocket-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

---

## 一句话说清楚

传统竞品分析需要分析师手动搜集信息、交叉对比、撰写报告，耗时 3-5 天。CompIntel Research 把这个流程自动化：**输入「分析 Notion 在协作工具市场的竞争格局」，2-3 分钟后拿到一份含 SWOT 矩阵和引用来源的结构化报告。**

---

## 架构概览

```
用户输入
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    CompIntel Research Pipeline (10 nodes)                  │
│                                                                          │
│  1. IntentAnalyst     ────  解析查询 → 识别竞品 + 市场赛道                   │
│         │                                                                │
│  2. ResearchPlanner   ────  生成「竞品 × 维度」研究计划矩阵                   │
│         │                                                                │
│  3. CompetitorProfiler                                                   │
│         ├── send() 扇出 ── 每个竞品 → 子图并行画像                          │
│         │                   ┌─────────────────────┐                      │
│         │                   │ Search ∥ Scrape ∥ RAG │  (BM25+Dense 混合)  │
│         │                   │       → 汇聚聚合       │                      │
│         │                   └─────────────────────┘                      │
│         │                                                                │
│  4. Curator           ────  清洗画像 + 证据质量分级 (rich/adequate/thin)    │
│         │                                                                │
│  5. MarketAnalyst     ────  汇总 → 市场格局 + 趋势 + 空白点 (+ 自检)        │
│         │                                                                │
│  6. SWOT Synthesizer  ────  逐竞品独立 SWOT + 横向交叉分析 (+ 自检)         │
│         │                                                                │
│  7. ReportWriter      ────  分段装配报告 (摘要/叙事/结论，4 次独立 LLM 调用)  │
│         │                                                                │
│  8. Editor            ────  统稿：术语统一 + 去重 + 矛盾检测 + 重写摘要       │
│         │                                                                │
│  9. Reviewer          ────  质量审核门控 (LLM-as-Judge + 规则回退)           │
│         │                                                                │
│         ▼                                                                │
│    审核通过 → 10. RAG Ingest (报告回写 Qdrant 自增长记忆) → 导出 Bundle      │
│    审核不通过 → 返回 ReportWriter 修订 (最多 3 次)                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## 产出物

每次分析生成一个 **交付文件包（Bundle）**，包含 4 个文件：

| 文件 | 内容 |
|------|------|
| `report.md` | 完整的竞品分析报告（执行摘要 → 竞品画像 → 市场分析 → SWOT 矩阵 → 审核反馈） |
| `progress.md` | 执行进度摘要（各阶段状态 + 事件流 + 检查点） |
| `snapshot.json` | 全量结构化数据快照（供下游系统消费） |
| `manifest.txt` | 交付清单（生成时间 + 文件索引） |

---

## 技术栈

| 层 | 技术 | 角色 |
|---|------|------|
| **Agent 编排** | LangGraph StateGraph + `send()` fan-out + SqliteSaver Checkpointer | 核心引擎 |
| **意图解析** | LLM (DeepSeek / Kimi / GLM / OpenAI-compatible) + 启发式 fallback | IntentAnalyst |
| **信息检索** | Tavily / SerpAPI 搜索 + 自适应改词重搜 (ReAct) / BeautifulSoup 抓取 / Qdrant BM25+Dense 混合检索 + RRF 融合 | CompetitorProfiler |
| **数据模型** | Pydantic v2 — 请求/响应/内部状态全量 Schema 约束 | 类型安全 |
| **API 服务** | FastAPI + WebSocket — 完成后事件回放 | 对外接口 |
| **前端界面** | Next.js + React + TypeScript + Tailwind | 本地演示 UI |
| **追踪审计** | JSONL 审计日志 + ExecutionTracker (checkpoint / decision / risk 全记录) | 可观测性 |
| **质量保障** | 7 维度离线评测 + 自动化回归测试 + 节点级自检 (Market/SWOT) | 评测体系 |
| **Prompt 管理** | YAML 注册表 + 版本化 + 独立参数 (model/temperature/max_tokens) | Prompt 工程 |
| **CLI** | `python -m compintel.run` | 本地入口 |

---

## 快速开始

### 环境要求

- Python 3.11+
- （可选）LLM API Key 与搜索 API Key —— 不配置也能跑，系统会使用启发式 fallback；真实搜索需要配置 `.env`

### 安装

```bash
# 完整依赖（含 CLI + API 服务 + 搜索/抓取/RAG）
pip install -r requirements.txt

# 可编辑安装（适合本地开发）
pip install -e .
```

### 30 秒跑通

```bash
# CLI 模式
python -m compintel.run "分析 Notion 在协作工具市场的主要竞品"

# 输出示例:
# Running CompIntel analysis...
# Bundle: outputs/compintel_bundle_20260617_130000_abc123
# Status: in_progress
```

### 启动 API 服务

```bash
uvicorn compintel.api:create_app --factory --reload
```

```
GET  /health                   → {"status": "ok"}
POST /api/compintel/analyze    → 竞品分析（JSON 响应）
WS   /ws/compintel             → 完成后事件回放（mode=replay）
```

> 当前 WebSocket 行为是回放模式：客户端发送 `{ "query": "..." }` 后，服务端先完成一次分析，再按顺序推送事件列表，并在最终 `analysis_ready` 消息中返回报告 bundle 路径。前端可用 `setTimeout` 将事件逐条展示为进度动画。

### 启动前端

```bash
cd frontend
npm install
npm run dev
```

默认连接 `http://localhost:8000/ws/compintel`。如果后端地址不同，可设置：

```bash
NEXT_PUBLIC_COMPINTEL_API_BASE=http://localhost:8000
```

### 运行测试

```bash
python -m pytest tests/test_compintel_core.py -q
# 45 passed ✓
```

---

## 项目结构

```
compintel/
├── agents/
│   ├── intent_analyst.py       # 意图解析（LLM + 启发式双路径）
│   ├── research_planner.py     # 研究规划
│   ├── competitor_profiler.py  # 竞品画像（Search/Scrape/RAG 三路聚合）
│   ├── search_worker.py        # 搜索 worker（自适应改词重搜 ReAct）
│   ├── scrape_worker.py        # 网页抓取 worker（分行业信源）
│   ├── rag_retriever.py        # RAG 检索 worker (BM25+Dense 混合)
│   ├── curator.py              # 画像清洗 + 证据质量分级
│   ├── market_analyst.py       # 市场格局分析 (+ 自检)
│   ├── swot_synthesizer.py     # SWOT 综合 + 交叉分析 (+ 自检)
│   ├── report_writer.py        # 分段装配报告
│   ├── editor.py               # 主编统稿（术语统一/去重/矛盾检测）
│   ├── reviewer.py             # 质量审核门控 (LLM-as-Judge + 规则回退)
│   └── base.py                 # Agent 基类
├── core/
│   ├── llm_service.py          # 统一 LLM 服务（调用/重试/解析/降级/推理压缩）
│   └── state_adapter.py        # 类型化 State 读写适配器
├── prompts/
│   ├── __init__.py             # Prompt 注册表 (load_prompt + 安全格式化)
│   ├── intent_analyst.yaml     # v1.0.0
│   ├── research_planner.yaml
│   ├── market_analyst.yaml
│   ├── swot_competitor.yaml
│   ├── swot_cross.yaml
│   ├── reviewer.yaml
│   ├── editor.yaml
│   └── curator.yaml
├── tools/
│   ├── __init__.py             # Tool 协议 (dataclass)
│   └── registry.py             # 可插拔数据源注册表 (3 tools)
├── rag/
│   ├── qdrant_store.py         # Qdrant 混合检索 (BM25+Dense+RRF)
│   └── data_loader.py          # 竞品种子数据预加载
├── graph.py                    # 10 节点 Pipeline 编排器
├── execution.py                # 执行封装（事件 + 追踪 + 审计）
├── schemas.py                  # Pydantic 数据契约
├── state.py                    # TypedDict 状态定义
├── evaluate.py                 # 7 维度离线评测器
├── regression_test.py          # 自动化回归测试流水线
├── tracker.py                  # 执行追踪器（checkpoint/decision/risk）
├── audit_store.py              # JSONL 审计日志
├── bundle.py                   # 交付文件包生成器
├── events.py                   # 事件类型定义
├── parsing.py                  # JSON 容错解析 (json_repair + 截断修复)
├── progress.py                 # 进度摘要格式化
├── settings.py                 # 环境变量配置
├── api.py                      # FastAPI 应用工厂
├── run.py                      # CLI 入口
└── server.py                   # 服务启动

frontend/
├── app/                        # Next.js App Router
├── components/                 # 输入、进度、SWOT、对比表、报告组件
└── lib/                        # WebSocket 客户端、类型、Markdown 转换

tests/
├── test_compintel_core.py      # 45 个核心测试（全链路 + LLM/fallback 双路径）
└── test_regression.py          # 回归测试（baseline diff + CI gate）
```

---

## 核心设计决策

### 1. 全 Agent 三层降级容错

每个 Agent 的降级路径因职责不同而异：

| Agent | LLM 路径 | 降级路径 |
|---|---|---|
| IntentAnalyst | `_try_llm_parse()` | 正则提取 → 关键词推断 → 种子竞品 → 模板问题 |
| ResearchPlanner | `_try_llm_plan()` | 模板 4 阶段计划 |
| MarketAnalyst | `_try_llm_analyze()` | `_derived_analysis()` (基于画像) → `_fallback_analysis()` (模板) |
| SWOT | `_try_llm_synthesize()` | `_derived_swot()` (基于画像) → `_fallback_swot()` (模板) |
| ReportWriter | section-by-section 装配 | 单段降级，其他段 LLM 质量保留 |
| Reviewer | `_try_llm_review()` | `_fallback_review()` (规则评分 + 模板/空数据检测 + 行业错配) |
| Editor | `_edit_report()` | 透传未编辑报告 (零阻塞) |

系统在**没有 API Key 的情况下也能给出有意义的结果**。

### 2. RAG 自增长记忆 + 混合检索

- **BM25 稀疏 + Dense 语义 + RRF 融合排序**：精确术语匹配 (如 "BYD" ↔ "比亚迪") 由 BM25 兜底，语义相似由 Dense Embedding 兜底，Qdrant Prefetch + RRF 自动融合
- **自增长闭环**：审核通过的报告自动写回 Qdrant，后续分析可检索历史洞察
- **零新增依赖 BM25**：纯 Python 实现，CJK bigram + ASCII word 分词，增量建词表

### 3. 搜索节点 ReAct 自适应

搜索 worker 内建质量评估 → 改词重搜循环：
- **自评估**（纯规则，不调 LLM）：检查结果数量、摘要深度、PR/新闻比例
- **自适应改词**：第 1 轮追加分析关键词，第 2 轮切换行业报告角度
- 最多 3 轮，始终保留最佳批次

### 4. 节点级自检 (非阻塞)

MarketAnalyst 和 SWOT Synthesizer 在 LLM 产出后进行纯规则自检（集合运算，<1ms）：
- 检测幻影公司名（不在输入 profile 中）
- 检测重复分类（同公司出现在 leaders + challengers）
- 检测竞品遗漏
- 发现问题追加 warnings，不阻塞管道

### 5. Prompt 注册表 + 版本化

所有 prompt 外置到 `compintel/prompts/*.yaml`，每个文件独立版本号，包含 `model_key` / `max_tokens` / `temperature` 参数。安全格式化器区分 `{var}` 占位符和 JSON `{"key"}` 字面量。缺失 YAML 文件时自动回退到内置默认。

### 6. 自动化回归评测

10 条代表性查询的离线评测集，7 维度评分，baseline JSON 快照：
```bash
python -m compintel.regression_test              # 跑全量 + diff
python -m compintel.regression_test --update-baseline  # 更新 baseline
```
CI 兼容：`test_regression.py` 提供 baseline 结构校验 + `@pytest.mark.slow` 全量回归。

### 7. 可审计的追踪系统

每次分析全程记录 checkpoint、decision、risk、pending question。每个 checkpoint 带 owner + evidence + summary。审计日志以 JSONL 格式持久化。

### 8. 交付文件包设计

分析结果以 bundle 形式输出 —— 报告、进度、原始数据、清单 —— 可被下游系统消费。

---

## 配置

项目支持用 `.env` 做本地配置。先复制模板：

```bash
cp .env.example .env
```

然后按需替换真实 key。`.env` 已加入 `.gitignore`，不会提交到 GitHub。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | `deepseek` | LLM 服务商。当前支持 `deepseek`、`kimi`、`glm`、`openai-compatible` |
| `LLM_API_KEY` | — | 统一 LLM API Key 字段 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI-compatible 服务地址 |
| `FAST_LLM` | `deepseek-chat` | 快速 Agent 使用的模型 |
| `SMART_LLM` | `deepseek-chat` | 智能 Agent 使用的模型 |
| `STRATEGIC_LLM` | `deepseek-reasoner` | 战略/推理 Agent 使用的模型 |
| `LLM_TIMEOUT_SECONDS` | `60` | 单次 LLM 请求超时秒数。超时后 Agent 自动降级 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh` | 本地 embedding 模型。留空则使用轻量 HashEmbedder |
| `HF_HOME` | `D:/huggingface` | HuggingFace 模型缓存目录 |
| `SEARCH_PROVIDER` | `tavily` | 搜索服务商。当前支持 `tavily`、`serpapi` |
| `SERPAPI_API_KEY` | — | 统一搜索 API Key 字段。使用 Tavily 时也填这里 |
| `COMPINTEL_AUDIT_PATH` | `outputs/compintel_audit.jsonl` | 审计日志路径 |

Qdrant RAG 支持两种运行方式：

- 开发/测试：默认使用 Qdrant `:memory:` 模式，无需 Docker。
- 持久化服务：安装 Docker 后运行 `docker run -p 6333:6333 qdrant/qdrant`。

**DeepSeek + Tavily 默认配置：**

```bash
LLM_PROVIDER=deepseek
LLM_API_KEY=replace-with-your-deepseek-api-key
LLM_BASE_URL=https://api.deepseek.com/v1
FAST_LLM=deepseek-chat
SMART_LLM=deepseek-chat
STRATEGIC_LLM=deepseek-reasoner

SEARCH_PROVIDER=tavily
SERPAPI_API_KEY=tvly-your_tavily_key_here
```

Kimi、GLM、SerpApi 的占位配置已写在 `.env.example`，需要时取消注释即可。

---

## Roadmap

| 阶段 | 目标 | 状态 |
|------|------|------|
| **Week 1** | 项目骨架 + Agent Pipeline + 意图解析 + 审计追踪 + 测试 + API | ✅ |
| **Week 2** | LangGraph StateGraph + `send()` 并行子图 + Search/Scrape/RAG Worker | ✅ |
| **Week 3** | LLM 全接入 + 三层降级 + 报告分段装配 + Curator/Editor/Reviewer | ✅ |
| **Week 4** | 前端 UI + WebSocket + RAG 回写闭环 + 评测器 | ✅ |
| **P0 增强** | BM25+Dense 混合检索 + 搜索 ReAct 化 + 自动化回归评测 | ✅ |
| **P1 增强** | 节点自检 + Prompt 注册表版本化 + Tool 协议抽象 | ✅ |
| **后续** | 语义分块 / 历史分析注入 / 意图复杂度分流 | ⬜ |

---

## 许可

MIT License
