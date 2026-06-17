# CompIntel Research

> **Competitive Intelligence Agent System** — 输入一个公司名，7 个 Agent 协作完成竞品画像、市场分析、SWOT 综合与审核报告。

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Week%201%20Scaffold-orange)](#roadmap)

---

## 一句话说清楚

传统竞品分析需要分析师手动搜集信息、交叉对比、撰写报告，耗时 3-5 天。CompIntel Research 把这个流程自动化：**输入「分析 Notion 在协作工具市场的竞争格局」，2-3 分钟后拿到一份含 SWOT 矩阵和引用来源的结构化报告。**

---

## 架构概览

```
用户输入
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CompIntel Research Pipeline                    │
│                                                                  │
│  1. IntentAnalyst     ────  解析查询 → 识别竞品 + 市场赛道         │
│         │                                                        │
│  2. ResearchPlanner   ────  生成「竞品 × 维度」研究计划矩阵         │
│         │                                                        │
│  3. CompetitorProfiler ────  并行画像（Search ∥ Scrape ∥ RAG）    │
│         │                                                        │
│  4. MarketAnalyst     ────  汇总 → 市场格局 + 趋势 + 空白点         │
│         │                                                        │
│  5. SWOT Synthesizer  ────  每个竞品 SWOT + 横向对比矩阵           │
│         │                                                        │
│  6. ReportWriter      ────  结构化竞品报告（含引用来源）             │
│         │                                                        │
│  7. Reviewer          ────  质量审核门控（LLM-as-Judge）            │
│         │                                                        │
│         ▼                                                        │
│    审核通过 → 导出 (Markdown / JSON / 审计日志)                     │
│    审核不通过 → 返回 ReportWriter 修订 (最多 3 次)                   │
└──────────────────────────────────────────────────────────────────┘
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
| **Agent 编排** | 当前：轻量 Pipeline → **Week 2 迁移至 LangGraph**（StateGraph + `send()` 并行 + Checkpointer 持久化） | 核心引擎 |
| **意图解析** | LLM (OpenAI-compatible / DeepSeek) + 启发式 fallback | IntentAnalyst |
| **信息检索** | Tavily Search API + BeautifulSoup 网页抓取 + Qdrant 向量检索（RAG） | CompetitorProfiler |
| **数据模型** | Pydantic v2（请求/响应/内部状态全量 Schema 约束） | 类型安全 |
| **API 服务** | FastAPI + WebSocket（实时事件流推送） | 对外接口 |
| **追踪审计** | JSONL 审计日志 + ExecutionTracker（checkpoint / decision / risk 全记录） | 可观测性 |
| **CLI** | `python -m compintel.run` | 本地调试 |
| **参考来源** | [GPT Researcher](https://github.com/assafelovic/gpt-researcher)（LangGraph 7-Agent 架构的行业标杆参考） | 架构灵感 |

---

## 快速开始

### 环境要求

- Python 3.11+
- （可选）OpenAI 兼容 API Key —— 不配置也能跑，使用启发式 fallback

### 安装

```bash
# 基础依赖
pip install pydantic

# 完整依赖（含 LLM 调用 + API 服务）
pip install pydantic fastapi uvicorn
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
pip install fastapi uvicorn
uvicorn compintel.api:create_app --factory --reload
```

```
GET  /health                   → {"status": "ok"}
POST /api/compintel/analyze    → 竞品分析（JSON 响应）
WS   /ws/compintel             → 实时事件流推送
```

### 运行测试

```bash
python -m pytest tests/test_compintel_core.py -q
# 7 passed ✓
```

---

## 项目结构

```
compintel/
├── agents/
│   ├── intent_analyst.py       # 意图解析（LLM + 启发式双路径）
│   ├── research_planner.py     # 研究规划
│   ├── competitor_profiler.py  # 竞品画像（Search/Scrape/RAG 三路聚合）
│   ├── market_analyst.py       # 市场格局分析
│   ├── swot_synthesizer.py     # SWOT 综合 + 对比矩阵
│   ├── report_writer.py        # 报告撰写
│   └── reviewer.py             # 质量审核门控
├── export/
│   └── formatter.py            # Markdown 报告格式化
├── rag/
│   ├── qdrant_store.py         # Qdrant 向量库封装
│   └── data_loader.py          # 竞品种子数据预加载
├── prompts/
│   └── templates.py            # 竞品分析专用 Prompt 模板族
├── graph.py                    # 7 节点 Pipeline 编排器
├── execution.py                # 执行封装（事件 + 追踪 + 审计）
├── schemas.py                  # Pydantic 数据契约（10+ Schema）
├── state.py                    # TypedDict 状态定义
├── tracker.py                  # 执行追踪器（checkpoint/decision/risk）
├── audit_store.py              # JSONL 审计日志
├── bundle.py                   # 交付文件包生成器
├── events.py                   # 事件类型定义
├── parsing.py                  # JSON 容错解析（json_repair 集成）
├── progress.py                 # 进度摘要格式化
├── settings.py                 # 环境变量配置
├── api.py                      # FastAPI 应用工厂
├── run.py                      # CLI 入口
└── server.py                   # 服务启动

tests/
└── test_compintel_core.py      # 7 个核心测试（追踪/格式化/事件/打包/API）
```

---

## 核心设计决策

### 1. 三层容错意图解析

IntentAnalyst 是系统的入口，我们做了三层降级保障：

```
LLM 调用（OpenAI/DeepSeek）
  │
  ├── 成功 → 解析 JSON → 返回结构化结果
  │
  └── 失败/未配置 API Key
        │
        ├── 正则表达式从查询中提取目标公司名
        ├── 关键词规则推断市场赛道
        ├── 种子生成候选竞品列表
        └── 模板生成研究问题
```

这意味着系统在**没有 API Key 的情况下也能给出有意义的结果**——这在面试场景中非常有价值，因为你永远不知道 Demo 现场的网络状况。

### 2. 可审计的追踪系统

每次分析全程记录 checkpoint、decision、risk、pending question。每个 checkpoint 带 owner + evidence + summary。审计日志以 JSONL 格式持久化，可直接用 `jq` 查询。

### 3. 交付文件包设计

不是只输出一段 Markdown 文本，而是输出一个**完整的分析包**——报告、进度、原始数据、清单。这意味着分析结果可以被下游系统（Notion、飞书文档、内部 BI）消费。

---

## 配置

通过环境变量配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FAST_LLM` | `openai:gpt-4o-mini` | 快速 Agent 使用的模型 |
| `SMART_LLM` | `openai:gpt-4o` | 智能 Agent 使用的模型 |
| `STRATEGIC_LLM` | `openai:gpt-4o` | 战略 Agent 使用的模型 |
| `OPENAI_BASE_URL` | — | 兼容 OpenAI API 的端点（DeepSeek 等） |
| `OPENAI_API_KEY` | — | API Key（不设则使用启发式 fallback） |
| `COMPINTEL_AUDIT_PATH` | `outputs/compintel_audit.jsonl` | 审计日志路径 |

**用 DeepSeek：**

```bash
export FAST_LLM="openai:deepseek-chat"
export SMART_LLM="openai:deepseek-chat"
export STRATEGIC_LLM="openai:deepseek-chat"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export OPENAI_API_KEY="sk-your-key"
```

---

## Roadmap

| 阶段 | 目标 | 状态 |
|------|------|------|
| **Week 1** | 项目骨架 + 7 Agent Pipeline + 意图解析 + 审计追踪 + 测试 + API | ✅ 完成 |
| **Week 2** | 迁移到 LangGraph（StateGraph + `send()` 并行子图 + Checkpointer） | 🔜 |
| **Week 3** | Qdrant RAG 集成 + 竞品文档知识库 + 混合检索（Dense + BM25） | 🔜 |
| **Week 4** | 前端竞品报告 UI（SWOT 可视化 + 对比矩阵）+ Demo 录制 | 🔜 |

---

## 面试准备

如果你也在用这个项目应聘 AI Agent 岗位，以下问题建议提前准备：

1. **「为什么选 LangGraph？」** → 有状态图编排 + `send()` 原生并行 + Checkpointer 断点续跑 + `interrupt` 人工审核
2. **「RAG 怎么设计的？」** → Qdrant HNSW + Dense(向量) + Sparse(BM25) + RRF 融合 + 按来源可信度/时效性重排序
3. **「怎么保证报告质量？」** → Reviewer LLM-as-Judge 评分（完整性/准确性/可操作性）+ 引用溯源校验 + 最多 3 次修订
4. **「最难的挑战？」** → 多源数据矛盾处理——实现了 Claim 提取层 + 来源可信度排序 + 呈现范围而非单一值

---

## 开发约定

- CompIntel Research 与本地参考目录 `gpt-researcher/` 独立（后者已加入 `.gitignore`）
- 运行时生成物（报告 / 审计日志 / bundle）写入 `outputs/`，不提交到仓库
- 优先沿用现有模块扩展能力，新增流程阶段时同步更新追踪/事件/测试
- 测试文件 `tests/` 中的用例保持可通过状态

---

## 许可

MIT License
