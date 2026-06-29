# CLAUDE.md

本文件约束 Claude Code 在 CompIntel Research 项目中的代码生成行为。
通用工程纪律参见 `~/.claude/CLAUDE.md`（Karpathy 10 条规则），本文档仅包含项目特有上下文。

---

## 项目上下文（必读）

### 技术栈
- **后端**: Python 3.11+, Pydantic v2, LangGraph (StateGraph + Send fan-out), FastAPI, httpx (AsyncClient)
- **向量存储**: Qdrant (BM25 + Dense + RRF hybrid retrieval), BAAI/bge-small-zh embedding
- **LLM 供应商**: DeepSeek / Kimi / GLM，OpenAI 兼容协议
- **前端**: Next.js 14.2 App Router, React 18, TypeScript, Tailwind CSS 3.4
- **测试**: pytest（45 核心测试 + regression test suite）

### 架构关键设计
- **三段降级**: 每个 Agent 都有 LLM → derived → template/fallback 三条路径
- **Qdrant 默认 :memory: 模式**，无需 Docker 即可开发
- **compintel/agents/**: 10 个 Agent，每个一个文件，通过 StateGraph 编排
- **compintel/core/**: LLM service（重试 + 3 级降级）、StateAdapter
- **compintel/prompts/**: YAML prompt 注册表，带版本号
- **compintel/rag/**: Qdrant hybrid store + data_loader
- **frontend/**: 单页 Next.js 应用，WebSocket 连接 FastAPI 后端

### 编码风格
- Python: snake_case, 类型注解 (TypedDict / Pydantic v2), docstring
- TypeScript: strict mode, 函数组件, Tailwind 设计 token (ink/panel/line)

### 项目特有约束
- 所有 LLM 调用必须走 `compintel/core/llm_service.py`，不要直接调 `httpx` 或 `openai`
- State 字段定义在 `compintel/state.py`——改动 state 会影响所有下游 agent
- 加新 agent：先实现 LLM 路径 + fallback 路径，不要加第四个降级层
- `requirements.txt` 是主依赖清单，项目不用 Poetry/PDM/Pipenv
- 前端类型定义在 `frontend/lib/types.ts`，前后端要对齐

### 项目特有翻车模式
- 改了 curator 顺手改 market_analyst 的 prompt → Kitchen Sink
- 第一个 agent 出现重复模式就想抽象到 core/ → 等至少 3 个再抽象
- 不确定 Qdrant/DeepSeek API 签名时 → 读项目源码，不要猜
- 改 graph.py pipeline 顺序导致连锁改动 → 超过 3 个文件时喊停

---

## 快速参考卡

| 动作 | 命令 |
|------|------|
| 跑测试 | `python -m pytest tests/ -q` |
| 端到端跑 | `python -m compintel.run "测试查询"` |
| 前端开发 | `cd frontend && npm run dev` |
| 后端 API | `python -m uvicorn compintel.api:app --reload --port 8000` |
| LLM 调用入口 | `compintel/core/llm_service.py` |
| State 定义 | `compintel/state.py` |
| 前端类型 | `frontend/lib/types.ts` |
| Pipeline 编排 | `compintel/graph.py` |
| Prompt 模板 | `compintel/prompts/*.yaml` |
| RAG 检索 | `compintel/rag/qdrant_store.py` |
