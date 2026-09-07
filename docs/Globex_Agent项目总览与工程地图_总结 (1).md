# Globex Agent 项目总览与工程地图

## 一、本章目标

本章用于建立 Globex 项目的整体认识，重点理解以下内容：

- Globex 是一个跨平台、对话式购物 Agent，而不是普通商品搜索框。
- 系统采用“1 个主 AgentLoop + N 个按需 fork 的同质子 AgentLoop”架构。
- 项目包含 9 个核心业务工具，以及向量召回、长期记忆、上下文压缩、AGUI 和评测训练等基础设施。
- `thread_id`、`session_dir`、ContextVar、HTTP 和 WebSocket 共同支撑前后端任务链路。
- 项目目录按照 Agent、API、工具、召回、记忆、压缩、评测等职责分层。

---

## 二、Globex 要解决的问题

### 1. 普通搜索方式的局限

普通电商助手通常采用以下链路：

```text
用户提问
  -> 将 query 直接传给搜索 API
  -> 返回商品列表
```

这种方式只能完成简单的关键词检索，难以处理以下复杂需求：

- 理解“便宜、抗造、小众、不要塑料”等隐含偏好。
- 同时查询多个电商平台并完成价格对比。
- 记住用户跨会话的长期偏好。
- 在信息不足时继续搜索、比较和筛选，而不是一次检索后直接结束。

### 2. Globex 的核心能力

Globex 的核心特点是：

```text
信息来源更多
+ 决策过程可以反复迭代
+ 支持跨会话长期记忆
+ 执行过程对前端可见
```

它接入的主要信息源包括：

| 信息来源 | 主要用途 | 项目承接方式 |
|---|---|---|
| 模型自身知识 | 通用品类知识、属性解释 | AgentLoop Think 阶段 |
| 跨平台商品数据 | 商品、价格、评分、运费 | ItemSearch、PriceCompare、ShippingCalc |
| 三塔向量召回 | 语义召回和个性化召回 | User / Query / Item 三塔 + ANN |
| 品类知识库 | 热卖商品、典型属性 | CategoryInsight + RAG |
| Web 实时资料 | 最新评测、趋势和推荐 | WebSearch |
| 用户长期偏好 | 材质、品牌、风格、黑名单 | 长期记忆 Store |

### 3. Globex 的完整购物链路

```text
理解用户购物意图
  -> Planner 拆解预算、品类和偏好
  -> 判断已有信息是否足够
  -> 必要时 fork 多个子 Agent
  -> 跨平台并行检索
  -> 汇总候选商品
  -> 比价并计算关税、运费
  -> 按用户偏好二次筛选
  -> 生成商品清单和购买理由
  -> 将新的用户偏好写入长期记忆
```

因此，Globex 更像一个会替用户做购物调研的研究员，而不是简单的搜索机器人。

---

## 三、整体架构

Globex 的整体架构可以概括为：

```text
1 个主 AgentLoop
+ N 个按需 fork 的同质子 AgentLoop
+ 9 个核心工具
+ 5 类基础设施
```

### 1. 主 AgentLoop

主 AgentLoop 是整个购物任务的统筹者，主要负责：

- 理解用户的原始购物需求。
- 使用 Planner 拆解预算、品类、材质、风格等子目标。
- 判断子任务由自己完成，还是 fork 子 Agent 执行。
- 收集子 Agent 返回的结构化结果。
- 继续执行比价、运费计算、商品筛选等步骤。
- 最终调用 ShoppingSummary 返回结果。
- 读写用户长期偏好。
- 向前端发送 AGUI 过程事件。

### 2. 同质子 AgentLoop

子 Agent 不是预定义的不同角色，而是主 AgentLoop fork 出来的完整克隆。

子 Agent 与主 Agent 具备：

- 相同的 system prompt。
- 相同的工具集。
- 相同的 Think → Act → Observe → Reflect 能力。

两者的主要区别如下：

| 维度 | 主 AgentLoop | 同质子 AgentLoop |
|---|---|---|
| `thread_id` | 用户主会话 ID | `sub-{uuid8}` |
| checkpoint | 主线对话历史 | 子任务独立历史 |
| 输入 | 用户原始 query | 主 Agent 分派的 `demands` |
| 输出 | 返回给前端 | 作为 `dispatch_tool` 的结果返回主 Agent |

对主 Agent 来说，子 Agent 的执行表现为一次普通工具调用，因此多 Agent 协作对主 AgentLoop 是透明的。

### 3. 什么时候需要 fork

主 AgentLoop 在 Think 阶段判断子任务是否满足以下任一条件：

| 判断条件 | 含义 | 示例 |
|---|---|---|
| 可以并行 | 多个任务相互独立 | 同时查询亚马逊、Shopee、速卖通、eBay |
| 需要上下文隔离 | 子任务数据量很大 | 单独分析大量候选商品 |
| 调用链较深 | 子任务内部需要多轮推理和工具调用 | 品类洞察需要依次分析爆款、属性和价格 |

若不满足以上条件，则由主 AgentLoop 自己处理。

### 4. `dispatch_tool`

`dispatch_tool(demands)` 是触发 fork 的元工具，不属于 9 个业务工具。

它的作用是：

```text
主 AgentLoop 调用 dispatch_tool
  -> 创建一个同质子 AgentLoop
  -> 子 Agent 独立完成 demands
  -> 将结果作为字符串返回主 AgentLoop
```

---

## 四、9 个核心工具

| 工具 | 主要调用者 | 作用 |
|---|---|---|
| `Planner` | 主 Agent | 将购物意图拆解为预算、品类、偏好和约束 |
| `ChatFallback` | 主 Agent | 处理闲聊或不需要检索的问题 |
| `WebSearch` | 主 Agent / 子 Agent | 搜索外部评测、趋势和实时资料 |
| `CategoryInsight` | 主 Agent / 子 Agent | 基于 RAG 获取品类爆款和属性洞察 |
| `ItemSearch` | 主 Agent / 子 Agent | 搜索单个平台的商品 |
| `ItemPicker` | 主 Agent / 子 Agent | 按用户偏好对候选商品二次筛选 |
| `PriceCompare` | 主 Agent | 对跨平台商品进行价格比较 |
| `ShippingCalc` | 主 Agent / 子 Agent | 估算关税和运费 |
| `ShoppingSummary` | 主 Agent | 生成最终商品清单和购买理由 |

其中，`ShoppingSummary` 是终结性工具，表示系统已经获得足够信息，可以生成最终结果。

---

## 五、基础设施体系

Globex 的基础设施可以归纳为五大类。

### 1. 向量召回与向量应用

- 使用 User、Query、Item 三塔模型完成跨语言、跨平台和个性化召回。
- 召回层使用 Faiss 的 HNSW + Inner Product。
- 生产环境可以进一步演进到 Milvus。
- OpenSearch 用于长期记忆和 RAG 商品知识库。
- 支持语义检索、全文检索和标量条件的混合查询。

### 2. 上下文压缩

使用 Cache Breakpoint 和自定义压缩策略：

- 防止长对话导致 token 数持续增长。
- 在压缩历史消息的同时尽量保持 Prompt Cache 命中率。
- 支撑 50 轮以上的长对话。

### 3. 长期记忆

通过 LangGraph BaseStore 接口和 OpenSearch 后端保存：

- 用户偏好。
- 材质或品牌黑名单。
- 历史选择。
- 跨会话购物习惯。

长期记忆会被注入主 AgentLoop 的 system prompt，使系统能够记住用户以前表达过的偏好。

### 4. AGUI 事件协议

AGUI 用于将 Agent 的执行过程实时展示给前端，例如：

- Planner 正在拆解需求。
- 子 Agent 正在执行跨平台检索。
- 工具调用开始或结束。
- 候选商品正在合流和比价。
- 最终结果已经生成。

### 5. 评测与训练闭环

项目通过以下方式持续优化模型行为：

```text
动态 Rubric
  -> 自动 Judge 评分
  -> 高分轨迹入库
  -> SFT 冷启动
  -> Agentic RL 继续训练
```

Rubric 采用 P0、P1、P2 等不同优先级描述每条 query 的评分要求。

---

## 六、AgentLoop 的运行循环

Globex 的核心循环是：

```text
Think
  -> Act
  -> Observe
  -> Reflect
  -> 再次 Think 或结束
```

具体流程如下：

```text
用户提交购物意图
  -> FastAPI 接收任务
  -> 注入 thread_id 和 session_dir
  -> 异步启动主 AgentLoop
  -> Think：判断下一步做什么
  -> Act：直接调用工具，或通过 dispatch_tool fork 子 Agent
  -> Observe：工具结果返回
  -> Reflect：判断信息是否足够
  -> 信息不足则继续循环
  -> 信息足够则调用 ShoppingSummary
  -> 将最终结果和过程事件返回前端
```

该循环允许 Agent 根据中间结果不断调整后续决策，而不是按照固定工作流一次性执行。

---

## 七、前后端交互方式

### 1. 为什么同时使用 HTTP 和 WebSocket

普通 HTTP 只能在任务全部完成后返回一次结果。

Globex 的任务可能包含多次工具调用、多个子 Agent 和跨平台检索，整个过程可能持续十几秒，因此采用：

```text
HTTP：启动任务
WebSocket：实时推送执行进度
```

典型交互流程：

```text
前端通过 HTTP 提交任务
  -> 后端立即创建任务并返回 thread_id
  -> 后台异步执行主 AgentLoop
  -> WebSocket 根据 thread_id 推送 AGUI 事件
  -> 任务结束后推送最终结果
```

### 2. `thread_id`

`thread_id` 表示当前会话或任务的身份，主要用于：

- 将执行进度推送到正确的前端连接。
- 隔离不同用户的 checkpoint。
- 管理 active_tasks。
- 取消指定任务。
- 路由 WebSocket 事件。

可以把它理解为：

```text
本次会话是谁
```

### 3. `session_dir`

`session_dir` 表示当前任务的工作目录，主要用于保存：

- 用户上传文件。
- 商品清单。
- 报告。
- Agent 生成的中间文件和最终文件。

可以把它理解为：

```text
本次任务的文件应该放在哪里
```

### 4. ContextVar 的作用

FastAPI 在收到请求后，将 `thread_id` 和 `session_dir` 写入 ContextVar。

之后，主 Agent、子 Agent 和工具可以在任意调用层级读取这些值，而不需要层层手动传参。

```text
FastAPI 创建上下文
  -> 主 AgentLoop 读取
  -> 工具读取
  -> fork 子 Agent 自动继承
  -> monitor 获取 thread_id
  -> 文件工具获取 session_dir
```

ContextVar 的核心价值是多用户异步任务隔离，避免不同任务之间发生串台。

---

## 八、关键 API 模块

| 文件 | 主要职责 | 一句话理解 |
|---|---|---|
| `app/api/context.py` | 保存 `thread_id` 和 `session_dir` | 我是谁，文件夹在哪 |
| `app/api/monitor.py` | 上报工具、fork 和任务结果事件 | 我现在正在做什么 |
| `app/api/connection.py` | 管理 `thread_id -> WebSocket` 连接 | 事件应该推给谁 |
| `app/api/server.py` | 提供 FastAPI 接口和任务入口 | 请求从哪里进入系统 |

工具内部只需要调用类似下面的方法：

```python
monitor.report_tool_start("item_search", ...)
```

工具本身不需要知道 WebSocket 如何连接，也不需要手动处理 `thread_id` 路由。

---

## 九、技术栈

| 层次 | 技术 | 作用 |
|---|---|---|
| Agent 范式 | AgentLoop | 主 Agent 和子 Agent 的循环执行 |
| Fork 机制 | `dispatch_tool(demands)` | 动态创建同质子 Agent |
| 模型接入 | LangChain、`init_chat_model` | 统一模型和工具调用接口 |
| 向量召回 | 三塔模型、Faiss、Milvus | 语义和个性化商品召回 |
| 向量应用 | OpenSearch | 混合检索、长期记忆和 RAG |
| 长期记忆 | LangGraph BaseStore | 跨会话保存用户偏好 |
| 上下文压缩 | Cache Breakpoint | 控制长对话 token |
| 事件协议 | AGUI | 实时展示 Agent 执行过程 |
| 后端 | FastAPI、Uvicorn、asyncio | 异步任务和接口服务 |
| 实时通信 | WebSocket | 按 `thread_id` 推送事件 |
| 前端 | React、Vite | 对话、商品卡片和事件可视化 |
| 评测 | Rubrics as Rewards | 动态生成评分标准 |
| 训练 | SFT、Agentic RL | 优化 Agent 行为 |
| 异步上下文 | ContextVar | 多用户任务隔离 |
| 路径管理 | pathlib、shutil | 管理上传和输出目录 |
| 环境配置 | python-dotenv | 读取 `.env` |
| 环境管理 | uv、Python 3.10 | 管理依赖和虚拟环境 |

---

## 十、项目目录结构

```text
globex-agent/
├── app/
│   ├── agent/       # 主 AgentLoop、模型、提示词和 fork 机制
│   ├── api/         # FastAPI、WebSocket、上下文和监控
│   ├── tools/       # 9 个核心 Agent Tool
│   ├── recall/      # 三塔模型和 ANN 召回
│   ├── memory/      # 长期记忆 Store
│   ├── compress/    # Cache Breakpoint 和上下文压缩
│   ├── eval/        # Rubric、Judge 和轨迹采集
│   ├── prompt/      # system prompt 和工具描述
│   └── utils/       # 路径、ContextVar 等普通 Python 工具
├── frontend/        # React + Vite 前端
├── docker/          # 本地依赖服务
├── examples/        # 前面章节的示例代码
├── tests/           # 工具、连接和任务测试
├── output/          # 运行时输出文件
├── uploaded/        # 用户上传文件
├── .env.example
├── .env
├── .python-version
├── pyproject.toml
└── uv.lock
```

各目录对应的职责如下：

| 目录 | 作用 |
|---|---|
| `app/agent/` | 组织主 AgentLoop 和 fork 子 Agent |
| `app/api/` | 接收请求、管理连接和推送事件 |
| `app/tools/` | 暴露给模型调用的业务工具 |
| `app/recall/` | 提供向量召回能力 |
| `app/memory/` | 保存和注入用户长期偏好 |
| `app/compress/` | 压缩长对话上下文 |
| `app/eval/` | 评测 Agent 执行轨迹 |
| `app/prompt/` | 管理提示词配置 |
| `app/utils/` | 提供普通 Python 辅助函数 |

---

## 十一、Agent Tool 与 Python Utils 的区别

### Agent Tool

Agent Tool 是暴露给模型调用的工具，例如：

```text
item_search
price_compare
shipping_calc
dispatch_tool
```

模型可以根据工具名称、描述和参数 Schema 主动选择调用它们。

### Python Utils

Python Utils 是后端代码内部使用的普通函数，例如：

```text
路径解析
ContextVar 封装
ANN 索引访问
长期记忆读写
```

这些能力不会直接暴露给模型。

例如：

```text
模型调用 item_search
  -> item_search 内部调用 app/recall/
  -> app/recall/ 完成三塔向量召回
  -> item_search 将结构化结果返回模型
```

模型只看到 `item_search` 的输入和输出，不需要知道内部的召回实现。

---

## 十二、依赖与环境准备

项目使用 uv 管理 Python 依赖和虚拟环境。

```bash
uv add -r requirements.txt
uv sync
```

命令作用：

| 命令 | 作用 |
|---|---|
| `uv add -r requirements.txt` | 将依赖写入 `pyproject.toml` 并更新 `uv.lock` |
| `uv sync` | 根据依赖声明创建或更新 `.venv` |

环境检查：

```bash
uv run python -V
uv run python -c "import langgraph, langchain, fastapi, faiss; print('ok')"
```

---

## 十三、核心链路总览

```text
用户提交购物需求
  -> FastAPI 创建 thread_id 和 session_dir
  -> ContextVar 保存当前任务上下文
  -> 主 AgentLoop 调用 Planner 拆解需求
  -> Think 阶段判断是否 fork
  -> dispatch_tool 创建同质子 Agent
  -> 子 Agent 跨平台调用 ItemSearch
  -> 三塔向量召回商品
  -> ShippingCalc 计算运费和关税
  -> 子 Agent 返回结果
  -> 主 Agent 汇总跨平台商品
  -> PriceCompare 完成比价
  -> ItemPicker 按偏好筛选
  -> ShoppingSummary 生成最终清单
  -> monitor 通过 WebSocket 推送 AGUI 事件
  -> 文件保存到 session_dir
  -> 用户偏好写入长期记忆 Store
```

---

## 十四、本章需要掌握的重点

1. Globex 是一个可规划、可检索、可比价、可记忆的跨平台购物 Agent。
2. 主 AgentLoop 负责统筹，同质子 AgentLoop 负责并行或隔离执行复杂子任务。
3. `dispatch_tool` 是 fork 子 Agent 的元工具，不属于业务工具。
4. fork 的判断依据是：可并行、需要上下文隔离、调用链较深。
5. 系统通过 Think → Act → Observe → Reflect 循环反复补充信息。
6. HTTP 用于启动任务，WebSocket 用于推送过程事件。
7. `thread_id` 负责身份、连接和 checkpoint 隔离。
8. `session_dir` 负责任务文件隔离。
9. ContextVar 让主 Agent、子 Agent 和工具能够透明获取任务上下文。
10. 项目目录按照 Agent、API、工具、召回、记忆、压缩和评测等职责分层。
11. Agent Tool 暴露给模型，Python Utils 只供后端代码内部调用。
12. 后续章节会将 `.env`、ContextVar、monitor、路径管理、LLM 和提示词配置落成可运行代码。
