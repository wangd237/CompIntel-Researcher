# CompIntel Research

CompIntel Research 是一个面向竞争情报研究的独立 Python 项目。它以 agent 协作流程为核心，把自然语言市场问题转化为结构化竞品画像、市场分析、SWOT 综合、带审核反馈的研究报告，以及可交付的分析文件包。

本项目不是 `gpt-researcher` 的产品改造版。当前目录中的 `gpt-researcher/` 仅作为本地参考资料保留，并已排除在版本控制之外。

## 当前能力

- 从研究问题输入到报告产出的端到端流程
- 意图分析：识别目标公司、市场赛道、竞品和研究问题
- 竞品研究规划与竞品画像生成
- 市场分析与 SWOT 综合
- 报告撰写与 reviewer 质量审核关卡
- 执行追踪：checkpoint、decision、risk、audit event 和进度摘要
- 交付文件包生成：报告、进度摘要、JSON 快照和 manifest
- FastAPI 与 WebSocket 集成入口
- 覆盖核心执行、追踪、格式化和打包逻辑的 pytest 测试

## 目录结构

```text
compintel/
  agents/            各研究阶段的 agent 实现
  export/            Markdown 报告格式化
  prompts/           Prompt 模板基础结构
  rag/               检索与上下文组装接口
  api.py             FastAPI 与 WebSocket 应用工厂
  audit_store.py     JSONL 审计日志存储
  bundle.py          交付文件包生成器
  events.py          运行时事件契约
  execution.py       高层执行封装
  graph.py           研究流程编排
  progress.py        进度摘要格式化
  schemas.py         Pydantic 请求与响应契约
  settings.py        环境变量配置
  state.py           共享状态契约
  tracker.py         执行追踪器
  run.py             CLI 入口
tests/
  test_compintel_core.py
```

运行时生成物默认写入 `outputs/`。

## 执行流程

当前流程由 `compintel.graph.CompIntelGraph` 组装：

1. `intent_analyst` 解析研究问题，提取目标、市场赛道、竞品和研究问题。
2. `research_planner` 基于意图生成分析计划。
3. `competitor_profiler` 生成结构化竞品画像。
4. `market_analyst` 汇总市场层面的分析信号。
5. `swot_synthesizer` 生成 SWOT 分析。
6. `report_writer` 生成报告内容。
7. `reviewer` 执行质量审核并输出反馈。

`compintel.execution.CompIntelExecution` 在流程外层补充审计日志、事件输出和 tracker 快照。`compintel.bundle.generate_delivery_bundle(...)` 负责生成最终交付文件包。

## 快速开始

建议使用 Python 3.11+。当前核心代码和测试需要：

```powershell
python -m pip install pydantic pytest
```

运行一次本地分析：

```powershell
python -m compintel.run "分析 Notion 在协作工具市场的主要竞品"
```

命令会输出生成的 bundle 路径。一个 bundle 通常包含：

- `report.md`
- `progress.md`
- `snapshot.json`
- `manifest.txt`

默认审计日志路径为 `outputs/compintel_audit.jsonl`。

## API 入口

`compintel.api.create_app()` 提供：

- `GET /health`
- `POST /api/compintel/analyze`
- `WS /ws/compintel`

FastAPI 在当前阶段是可选依赖。运行 API 前先安装：

```powershell
python -m pip install fastapi uvicorn
uvicorn compintel.api:create_app --factory --reload
```

## 配置项

`CompIntelSettings` 会读取以下环境变量：

- `FAST_LLM`
- `SMART_LLM`
- `STRATEGIC_LLM`
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `COMPINTEL_AUDIT_PATH`

其中 `COMPINTEL_AUDIT_PATH` 默认值为 `outputs/compintel_audit.jsonl`。

## 测试

运行核心测试：

```powershell
python -m pytest tests/test_compintel_core.py -q
```

可选的语法检查：

```powershell
python -m compileall compintel tests
```

## 开发约定

- 保持 CompIntel Research 与本地参考目录 `gpt-researcher/` 独立。
- 优先沿用现有的 agent、graph、schema、tracker 和 bundle 模块扩展能力。
- 运行输出、审计日志和交付 bundle 保持在 `outputs/`，不提交到仓库。
- 新增流程阶段时，同步更新编排逻辑、事件/追踪覆盖和 README。
