# CLAUDE.md

本文件约束 Claude Code 在 CompIntel Research 项目中的代码生成行为。  
规则不是建议，是纪律。遵循则产出无需重写的代码，忽视则产出看似漂亮、生产环境必炸的代码。

---

## 0. 项目上下文（必读）

### 技术栈
- **后端**: Python 3.11+, Pydantic v2, LangGraph (StateGraph + Send fan-out), FastAPI, httpx (AsyncClient)
- **向量存储**: Qdrant (BM25 + Dense + RRF hybrid retrieval), BAAI/bge-small-zh embedding
- **LLM 供应商**: DeepSeek / Kimi / GLM，OpenAI 兼容协议
- **前端**: Next.js 14.2 App Router, React 18, TypeScript, Tailwind CSS 3.4
- **测试**: pytest（45+ 核心测试 + regression test suite）

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
- 文档: 中英双语

---

## 1. 先读再写

**改代码前，必须读懂你要改的文件及其上下文。**

- 读目标文件的完整 import 区域——它们告诉你项目在用什么。本项目用 `httpx` 不是 `requests`，用 `Pydantic v2` 不是 v1
- 看同一目录下至少一个同类文件——compintel/agents/ 下每个 agent 的模式都一样（继承 BaseCompIntelAgent, 实现 `__call__` 或 `execute`）
- 检查 `compintel/core/llm_service.py`——所有 LLM 调用必须走这个 service，别直接调 `httpx` 或 `openai`
- 读 `compintel/state.py`——TypedDict state 定义了所有 agent 间的数据合约
- 检查 `frontend/lib/types.ts`——前后端类型要对齐

**不确定就说。**"我在这个项目里没看到 X 的模式" 比凭空猜测好一个数量级。

---

## 2. 想清楚再写

**整理需求，对齐方案，再碰代码。**

- 需求模糊时，先说出你的理解和假设。"我理解你要给 reviewer 加一个重试参数，放在 CompIntelSettings 里，默认值 3"
- 如果改动可能影响状态流转（TypedDict state 字段），先标注——"这个字段需要在 state.py 里加，会影响所有下游 agent 的类型签名"
- 针对本项目：改动 research pipeline 时，想清楚是改单个 agent 还是改 graph 编排 (graph.py)；前者通常只需改一个文件，后者可能涉及 fan-out 逻辑和 checkpointer
- 前端性能问题先考虑：是 WebSocket 事件太多 → 合并？还是渲染太慢 → 虚拟化？不要猜，用 React DevTools Profiler 看

---

## 3. 极简

**写刚好解决当前问题的最少代码。不写「以后可能需要」的代码。**

- 本项目已有三段降级机制。加新 agent 时先实现 LLM 路径和 fallback 路径，不要加第四个降级层
- 加新功能到 agent 时：优先用 BaseCompIntelAgent 已有方法，优先复用 `llm_service.py`，不要在 agent 里重新发明 LLM 调用
- 前端：用已有的 Tailwind token (ink/panel/line)，不要新建颜色。组件已有 CompIntelInput、SWOTMatrix 等模式——新增 UI 时复用这个模式
- 对 compintel/prompts/ 下的 YAML 提示词：只加当前任务需要的字段，不要预加"未来可能需要的模板变量"
- **Copy-paste twice before you abstract.** 在 agent 里看到重复逻辑时，先容忍两次，第三次出现时再抽象到 `core/` 或 `tools/`

---

## 4. 精准修改

**Diff 越小越好。每行改动都是潜在的 bug、review 负担、git blame 噪音。**

- 修一个 agent 的 bug 时，不要顺便改另一个 agent 的变量名
- 匹配已有文件风格：Python 文件用单引号 vs 双引号，TypeScript 文件用分号 vs 无分号——跟原文件走
- 你的改动让某个 import / 变量 / 函数不再使用 → 删除它。但只删你自己的改动造成的。之前就存在的 dead code 不是这次的事
- **别格式化。** 不要对没格式化过的文件跑 black/prettier。格式化噪音会淹没你的实际改动
- 改完后看 diff：每一行都能直接关联到任务吗？"顺手改的" → revert

---

## 5. 验证

**能用的代码 vs 你以为能用的代码，区别是测试。**

- **修 bug 必须先写复现测试。** 这不是 TDD 教条，是唯一证明你修复了的办法
- 修改前后跑 `python -m pytest tests/ -q`。如果之前就有失败的测试，说出来，别让你的改动被冤枉
- 改 agent pipeline 后跑 `python -m compintel.run "测试查询"` 端到端
- 改前端组件后，确认 `npm run dev` 能正常渲染 + WebSocket 连接正常
- 本项目测试覆盖 agent 的 LLM 路径和 fallback 路径——你加的功能两条路径都要测
- **别写废话测试。** 测行为不是测实现。测有趣的边界（空输入、超长输入、并发），不是测 constructor

---

## 6. 目标驱动

**开始写代码前，先定义什么叫「做完」。**

| 模糊任务 | 精确化 |
|---------|-------|
| 给 curator 加验证 | "当 evidence 字段为空或不含 source_url 时，返回 grade=empty + 具体原因；为两种失败情况写测试" |
| 优化市场分析师 | "先 profile market_analyst 的一次调用耗时，找到瓶颈，只修那一个，修完对比耗时" |
| 加前端 loading 状态 | "在 CompIntelInput 提交后到第一个 WebSocket 事件到达前展示 spinner；type='thinking' 时展示脉冲动画" |

**多步骤任务必须先列计划：**

```
Plan:
1. 在 state.py 新增字段
2. 改相关 agent 的类型签名
3. 实现 agent 逻辑
4. 更新 prompt YAML
5. 写测试
6. 跑全量测试
```

用户可以在实现前纠正你的方向——这时候成本是 30 秒而不是 30 个文件。

---

## 7. 调试规范

**出问题时不猜测，调查。**

- 读完整个错误信息，包括 stack trace。Python traceback 精确告诉你哪行出了什么问题
- 先复现。改一行，测一行，确认修复。不要同时改三处然后不知道哪个修好的
- 对于 agent pipeline 问题：用 `run-logs/` 下的执行日志追踪具体哪个 agent 出问题，不要无脑改 prompt
- 不理解根因不加 workaround。某字段 null → 查为什么 null → 修复上游，而不是在接收方加 null check
- **如果卡住了，说。** "试了 X 和 Y，效果都不行。现象是 [...]，我怀疑是 Z" 比默默试 20 次有价值一万倍

---

## 8. 依赖管理

**加依赖是引入你不掌控的代码，它永久成为项目一部分。**

决策链路（严格按顺序）:
1. **项目已有的库能不能做？** 已有 httpx → 不加 requests。已有 Qdrant → 不加 ChromaDB。已有 Tailwind → 不加 CSS modules
2. **标准库能不能做？** Python `json` 模块 → 不加额外 JSON 库。`pathlib` → 不加 `os.path` 包装
3. **真的要加？** 检查：还在维护吗？多大？跟已有依赖冲突吗？

**本项目关键约束：**
- `requirements.txt` 是主依赖清单。不用 Poetry/PDM/Pipenv（项目只用 pip）
- 前端：`package.json` 已包含所有依赖，加之前检查是否已有
- 加依赖时必须说明原因："加 `tenacity` 因为 httpx 重试逻辑需要在 LLM service 层统一管理，标准库不提供指数退避"

---

## 9. 沟通规范

**代码之外的沟通质量同样决定协作效率。**

- **说清楚做了什么和为什么。** 不要只 dump 一段 diff。"我把 curator 的 evidence 字段从 str 改成 Optional[str]，因为 scrape_worker 失败时返回 None 会导致 curator 崩溃。这在 3 个下游 agent 里也需要适配"
- **主动标记疑虑。** "这个实现可以运行，但给 10 个竞品做 SWOT 是串行的，深度模式下可能需要 3 分钟。需要我改成并行吗？"
- **精准的不确定。** "我不确定 Qdrant 的 :memory: 模式是否支持 RRF 混合检索" ← 有用。"我觉得应该可以" ← 没用
- **匹配用户的知识水平。** 跟你讨论 compintel 架构时，不需要解释什么是 StateGraph。只有你引入的新概念才解释
- **Commit message 模板**: `[动作] [模块] when [触发条件]`。例: `fix curator evidence grading when scrape returns empty body`

---

## 10. 避坑清单

本项目最常出现的 7 种翻车模式。认出名字就要喊停：

| # | 模式 | 本项目典型表现 | 喊停方法 |
|---|------|-------------|---------|
| 1 | **Kitchen Sink** | 修 curator 的 evidence 字段，顺手把 market_analyst 的 prompt 也改了，顺便重命名了几个变量 | "我只被要求修 curator。其他改动单独提" |
| 2 | **Wrong Abstraction** | 第一个 agent 加了 `_call_llm_with_retry` 方法，立刻就想到要抽象成 `core/` 下的通用类 | 等至少 3 个 agent 出现同样模式再抽象 |
| 3 | **Invisible Decision** | 默默把 Qdrant 从 `:memory:` 改成磁盘模式，没标注这是个不可逆的选择 | "我改动了一个架构选择：Qdrant 从内存切到磁盘。这影响启动流程和数据持久化。" |
| 4 | **Optimistic Path** | 假设 Tavily API 总是返回结果，没处理 rate limit / 空结果 / JSON 解析失败 | Tavily 已经有了 fallback (SerpAPI)。加新数据源时也要考虑失败路径 |
| 5 | **Knowledge Hallucination** | 用 `qdrant_client.search()` 而不是项目里实际的 `qdrant_store.search_hybrid()` | 不确定 API 签名时，读源码。不要猜。本项目 `qdrant_store.py` 是唯一真相 |
| 6 | **Style Drift** | 在 TypedDict 为主的项目里引入 dataclass，在 TypeScript strict 项目里引入 `any` | 一致性 > 你的偏好。新文件跟着已有文件的风格走 |
| 7 | **Runaway Refactor** | 改 graph.py 的 pipeline 顺序 → 发现所有 agent 的 state key 都要改 → 又发现 prompt YAML 也要改 → 30 分钟过去了 | 连锁超过 3 个文件时："改动的范围在扩大——graph.py、state.py、至少 4 个 agent、prompt YAML。要继续还是缩小范围？" |

---

## 快速参考卡

你可以用以下命令来快速对照这些规则：

| 动作 | 对应命令 |
|------|---------|
| 跑测试 | `python -m pytest tests/ -q` |
| 端到端跑 | `python -m compintel.run "测试查询"` |
| 前端开发 | `cd frontend && npm run dev` |
| 后端 API | `python -m uvicorn compintel.api:app --reload --port 8000` |
| 检查 LLM 调用 | 读 `compintel/core/llm_service.py`（不要直接调 httpx/openai） |
| 检查 state 定义 | 读 `compintel/state.py` |
| 检查前端类型 | 读 `frontend/lib/types.ts` |
