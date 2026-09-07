# 16-3 可观测性体系与 LangFuse 全链路 Trace

**本章课程目标：**
- 理解为什么 Agent 比普通 Web 服务更需要可观测性——链路长、非确定性、token 成本不可预知。
- 掌握 LangFuse 接入 LangGraph 的四步流程：SDK → Trace → Span → Score。
- 理解 Trace 里应该记录的四类信息：工具调用链路 / Token 归因 / AgentLoop 轮次时间线 / RaR Score。
- 拿到一份"5 分钟定位 badcase"的 SOP——从 Score 筛选到 Span 定位到根因确认。
- 掌握工具 RT 告警和监控看板的指标清单。

**学习建议：** 这一章解决"线上出问题怎么快速找到根因"。第 8 章讲了 Rubric 评测（离线批量跑），本章讲的是"线上单条请求出了问题，5 分钟内定位到是哪一步、哪个工具、什么原因"——两者是互补关系。

**对应代码分支：**`16-3-langfuse-observability`

---

## 1、为什么 Agent 比普通 Web 更需要可观测性

### 1.1 普通 Web 的可观测性

普通 Web 服务出问题，通常是：
```text
请求进来 → 业务逻辑 → 数据库查询 → 返回

```

链路短、确定性高。出问题看 access log + error log + DB slow query 基本能定位。

### 1.2 Agent 的可观测性难题

Globex 一条请求的链路：
```text
请求进来
  → Think 1（LLM 推理，不确定输出什么）
  → Act 1（调 Planner，确定性）
  → Think 2（LLM 推理，可能决定 fork 也可能不 fork）
  → Act 2（fork 4 个子 Agent，每个内部再 Think → Act → ...）
  → Think 3（LLM 推理，基于上面所有结果做下一步决策）
  → ...
  → Act N（ShoppingSummary）
  → 返回

```

三个根本性困难：

| 困难  | 具体表现  |
|-------|-------------|
| **链路长**  | 5-10 轮 Think + Act，每轮都可能出问题  |
| **非确定性**  | 同一条 query 跑两次，工具调用顺序可能不同  |
| **token 成本不可预知**  | 一条请求可能花 5K token，也可能花 50K token，取决于模型决策  |

传统 log 完全不够用——你需要的是**结构化的 Trace 树**，能按请求展开每一步的输入输出、耗时、token 消耗。

### 1.3 LangFuse 解决什么

LangFuse 是为 LLM 应用设计的可观测性平台，核心概念：

| 概念  | 含义  | 在 Globex 里对应什么  |
|-------|-------|---------------------------|
| Trace  | 一次完整的用户请求  | 一条购物 query 的完整生命周期  |
| Span  | Trace 内的一个操作  | 一次 LLM 调用 / 一次工具调用  |
| Generation  | Span 的子类型，专指 LLM 调用  | Think 阶段的每次推理  |
| Score  | 对 Trace 的评分  | Rubric 评测分注入  |

---

## 2、LangFuse 接入四步

### 2.1 Step 1：安装 SDK
```bash
pip install langfuse

```

环境变量：
```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # 或自部署地址

```

### 2.2 Step 2：在 Agent 入口创建 Trace
```python
# app/api/server.py（修改 run_agent 入口）
from langfuse import Langfuse

langfuse = Langfuse()


async def run_agent(query: str, thread_id: str, user_id: str | None = None) -> dict:
    # 创建 Trace
    trace = langfuse.trace(
        name="globex-agent",
        id=thread_id,
        user_id=user_id,
        input={"query": query},
        metadata={"model": os.environ["LLM_MAIN"]},
    )

    # 把 trace 传递给后续调用（通过 ContextVar 或直接传参）
    set_langfuse_trace(trace)

    try:
        result = await _run_agent_internal(query, thread_id, user_id)
        trace.update(output={"final_answer": result.get("final", "")})
        return result
    except Exception as e:
        trace.update(output={"error": str(e)}, level="ERROR")
        raise
    finally:
        langfuse.flush()

```

### 2.3 Step 3：在工具调用处创建 Span
```python
# app/api/monitor.py（扩展 Monitor 类）
from app.observability.trace_ctx import get_langfuse_trace


class Monitor:
    async def report_tool_start(self, tool_name: str, args: dict) -> None:
        # 原有的 AGUI 推送逻辑不变...
        await self._emit("tool_start", f"正在调用 {tool_name}", {...})

        # 新增：LangFuse Span
        trace = get_langfuse_trace()
        if trace:
            span = trace.span(
                name=f"tool:{tool_name}",
                input=args,
                metadata={"thread_id": get_thread_id()},
            )
            set_current_span(span)  # 存到 ContextVar，tool_end 时关闭

    async def report_tool_end(self, tool_name: str, duration_ms: int) -> None:
        await self._emit("tool_end", f"{tool_name} 完成", {...})

        span = get_current_span()
        if span:
            span.end(
                output={"duration_ms": duration_ms},
                metadata={"tool_name": tool_name},
            )

```

### 2.4 Step 4：Rubric 评测后注入 Score
```python
# app/eval/rubric.py（评测完成后）
async def evaluate_and_score(trajectory: dict, trace_id: str):
    score = await evaluate_trajectory(trajectory)

    # 注入 LangFuse Score
    langfuse.score(
        trace_id=trace_id,
        name="rubric_total",
        value=score["total"] / 100.0,  # 归一化到 0-1
        comment=f"P0={score['p0']} P1={score['p1']} P2={score['p2']}",
    )

```

注入 Score 后，在 LangFuse 界面可以按分数筛选——直接点"Score < 0.65"就能看到所有低分 Trace。

---

## 3、Trace 里应该记录的四类信息

### 3.1 工具调用链路

每次 Act 阶段的完整记录：
```text
Span: tool:planner
  input: {"query": "旅行三件套 预算300 不要塑料"}
  output: {"budget": 300, "category": "旅行三件套", ...}
  duration: 720ms

Span: tool:dispatch_tool
  input: {"demands": "在 amazon 搜旅行三件套..."}
  output: "[{item_id: A01, ...}, ...]"
  duration: 1658ms
  metadata: {"sub_thread_id": "sub-9e1f-d1"}

```

**核心价值**：一眼看到"哪个工具花了多长时间、返回了什么"。

### 3.2 Token 归因

每次 LLM Generation 的 token 细分：
```text
Generation: think-1
  prompt_tokens: 3200
  completion_tokens: 180
  total_tokens: 3380
  cache_hit: true        ← 关键！说明 Prompt Cache 命中
  model: globex-main
  duration: 800ms

```

**核心价值**：
- 看 `cache_hit` 比例判断 Cache Breakpoint 是否工作正常
- 看 `prompt_tokens` 趋势判断上下文是否在膨胀
- 按 Trace 累加 total_tokens 看单次请求的成本

### 3.3 AgentLoop 轮次时间线

每轮 Think → Act → Observe → Reflect 的时间戳：
```text
Round 1: Think(800ms) → Act:planner(720ms) → Observe(5ms)
Round 2: Think(600ms) → Act:dispatch_tool×4(1658ms) → Observe(10ms)
Round 3: Think(500ms) → Act:price_compare(126ms) → Observe(5ms)
Round 4: Think(500ms) → Act:shipping_calc(38ms) → Observe(5ms)
Round 5: Think(500ms) → Act:item_picker(123ms) → Observe(5ms)
Round 6: Think(500ms) → Act:shopping_summary(2572ms) → Observe(5ms)
Total: 6 rounds, 8.7s

```

**核心价值**：定位延迟瓶颈在哪一轮、哪一步。

### 3.4 RaR Score

评测分直接挂在 Trace 上：
```text
Score: rubric_total = 0.79
  P0: pass
  P1: -2 (效率约束违反：item_search 重复调用 3 次)
  P2: quality=4/5, insight=4/5, decision=3/5

```

**核心价值**：按分数筛选低分 Trace，直接看到"为什么扣分"。

---

## 4、LangFuse Callback Handler 接入 LangGraph

### 4.1 完整实现
```python
# app/observability/langfuse_handler.py
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from app.observability.trace_ctx import get_langfuse_trace


def create_langfuse_handler(thread_id: str) -> LangfuseCallbackHandler | None:
    """为当前请求创建 LangFuse callback handler。

    接入方式：传给 LangGraph 的 config["callbacks"]。
    """
    trace = get_langfuse_trace()
    if not trace:
        return None

    return LangfuseCallbackHandler(
        trace_id=thread_id,
        session_id=thread_id,
        # LangFuse SDK 会自动记录每次 LLM 调用的 token / 延迟
    )

```

### 4.2 在 run_agent 里注入
```python
# app/agent/main_agent.py
from app.observability.langfuse_handler import create_langfuse_handler


async def run_agent(query: str, thread_id: str, user_id: str | None = None) -> dict:
    # ... 省略 Store 读取、prompt 拼装 ...

    handler = create_langfuse_handler(thread_id)

    callbacks = [handler] if handler else [ ]


    result = await asyncio.wait_for(
        agent.ainvoke(
            {"messages": [("user", query)]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": MAIN_AGENT_MAX_ITERATIONS,
                "callbacks": callbacks,  # ← 关键：注入 LangFuse
            },
        ),
        timeout=MAIN_AGENT_TIMEOUT_SEC,
    )
    # ...

```

LangFuse SDK 的 callback handler 会自动：
- 为每次 `llm.ainvoke()` 创建 Generation span
- 记录 prompt_tokens / completion_tokens / model / duration
- 为每次 tool 调用创建 Span

你在 Monitor 里额外记录的信息（如 AGUI 事件、fork 事件）是**补充**——LangFuse callback 解决基础面，Monitor 补充业务面。

---

## 5、5 分钟定位 badcase 的 SOP

### 5.1 完整流程
```text
Step 1（1 分钟）：打开 LangFuse 面板 → 按 Score < 0.65 筛选低分 Trace
Step 2（1 分钟）：选一条 Trace → 看 Score 的 comment 字段 → 知道扣分原因
Step 3（2 分钟）：展开 Trace 树 → 按时间线找到出问题的 Span
Step 4（1 分钟）：看 Span 的 input/output → 确认根因

```

### 5.2 三类典型 badcase 的定位路径

**Case A：工具返回为空**
```text
Score comment: "P1 违规：ShoppingSummary 内容为空"
  → 展开 Trace → 找到 shopping_summary Span
  → 看 input：picks 为空列表
  → 往上追：item_picker Span 的 output 也是空
  → 再往上追：item_search Span 返回 0 条
  → 根因：用户 query 太模糊，三塔召回命中 0 条
  → 修复：加兜底逻辑，召回为空时 fallback 到 WebSearch

```

**Case B：Cache 命中率骤降**
```text
监控告警：单 Session cache_hit 率从 80% 降到 20%
  → 按 thread_id 查 Trace
  → 看每个 Generation 的 cache_hit 字段
  → 发现从第 3 轮开始全部 miss
  → 看第 3 轮的 prompt_tokens：12000 → 18000（暴涨）
  → 根因：dispatch_tool 返回了 6000 token 的商品列表，撑破了压缩边界
  → 修复：调大 COMPRESS_KEEP_RECENT 或加工具结果截断

```

**Case C：延迟超标**
```text
监控告警：thread-xxx 总延迟 25s（阈值 12s）
  → 按 thread_id 查 Trace
  → 看轮次时间线：Round 2 的 dispatch_tool 花了 15s
  → 展开 Round 2：4 个子 Agent 中有 1 个超时（aliexpress 平台 API 响应慢）
  → 根因：aliexpress API P99 飙到 12s
  → 修复：该平台工具触发熔断（16-5 章）

```

---

## 6、工具 RT 告警

### 6.1 告警规则
```python
# app/observability/alerts.py
from dataclasses import dataclass


@dataclass
class AlertRule:
    tool_name: str
    p99_threshold_ms: int
    window_minutes: int = 5
    min_samples: int = 10


ALERT_RULES = [
    AlertRule("item_search", p99_threshold_ms=3000),
    AlertRule("price_compare", p99_threshold_ms=500),
    AlertRule("shipping_calc", p99_threshold_ms=200),
    AlertRule("category_insight", p99_threshold_ms=2000),
    AlertRule("shopping_summary", p99_threshold_ms=4000),
    AlertRule("dispatch_tool", p99_threshold_ms=5000),
]

```

### 6.2 告警触发逻辑
```python
# app/observability/alerts.py（续）
import numpy as np
from collections import deque


class ToolRTMonitor:
    def __init__(self):
        self._windows: dict[str, deque] = {}
        for rule in ALERT_RULES:
            self._windows[rule.tool_name] = deque(maxlen=200)

    def record(self, tool_name: str, duration_ms: int):
        if tool_name in self._windows:
            self._windows[tool_name].append(duration_ms)

    def check_alerts(self) -> list[str]:

        alerts = [ ]

        for rule in ALERT_RULES:
            window = self._windows[rule.tool_name]
            if len(window) < rule.min_samples:
                continue
            p99 = float(np.percentile(list(window), 99))
            if p99 > rule.p99_threshold_ms:
                alerts.append(
                    f"[ALERT] {rule.tool_name} P99={p99:.0f}ms > {rule.p99_threshold_ms}ms"
                )
        return alerts


rt_monitor = ToolRTMonitor()

```

### 6.3 告警通知

告警触发后通过 Webhook 推送到钉钉 / Slack：
```python
async def send_alert(message: str):
    import httpx
    webhook_url = os.environ.get("ALERT_WEBHOOK_URL")
    if not webhook_url:
        return
    async with httpx.AsyncClient() as client:
        await client.post(webhook_url, json={
            "msgtype": "text",
            "text": {"content": f"[Globex Agent] {message}"},
        })

```

---

## 7、和第 8 章 Rubric 评测的配合

### 7.1 两者的互补关系

| 维度  | Rubric 评测（第 8 章）  | LangFuse 可观测性（本章）  |
|-------|-----------------------------|----------------------------------|
| 运行时机  | 离线批量跑  | 线上实时  |
| 粒度  | 整条 Trace 的最终质量  | 每一步 Span 的延迟 / token / 结果  |
| 发现问题  | "这条 query 整体效果不好"  | "是第 3 轮 item_search 返回为空导致的"  |
| 修复指导  | 知道"该修"但不知道"改哪"  | 精确定位到具体 Span 和原因  |

### 7.2 Score 注入让两者联通

Rubric 评测跑完后把分数注入 LangFuse Trace，就能在 LangFuse 面板里：
- 按 Score 筛低分 Trace → 展开看具体 Span → 定位根因
- 按 Score 趋势看整体质量变化 → 发现模型升级 / 工具异常导致的批量退化

---

## 8、监控看板指标清单

### 8.1 推荐指标

| 类别  | 指标  | 告警阈值  | 数据源  |
|-------|-------|-------------|----------|
| **延迟**  | 总请求 P50 / P95 / P99  | P99 > 15s  | LangFuse Trace  |
| **延迟**  | 单工具 P99  | 各工具独立阈值  | LangFuse Span  |
| **Token**  | 单请求平均 token 消耗  | > 30K  | LangFuse Generation  |
| **Token**  | Cache hit 率  | < 60%  | LangFuse Generation  |
| **质量**  | Rubric 日均分  | < 0.70  | LangFuse Score  |
| **质量**  | 任务完成率  | < 85%  | 自定义统计  |
| **成本**  | 日 token 总消耗  | 超预算 120%  | LangFuse  |
| **稳定性**  | 降级率  | > 20%  | 16-4 章打标  |
| **稳定性**  | 熔断触发次数  | > 5 次/小时  | 16-5 章打标  |
| **错误**  | 5xx / timeout / 异常率  | > 2%  | FastAPI 中间件  |

### 8.2 看板布局建议
```text
┌──────────────────────────────────────────────────┐
│ Row 1: 核心 KPI                                   │
│  [总请求 P95] [任务完成率] [Rubric 日均分] [日成本] │
├──────────────────────────────────────────────────┤
│ Row 2: 延迟分布                                   │
│  [各工具 P99 柱状图]  [总延迟趋势线]               │
├──────────────────────────────────────────────────┤
│ Row 3: Token 与缓存                               │
│  [单请求 token 分布]  [Cache hit 率趋势]           │
├──────────────────────────────────────────────────┤
│ Row 4: 稳定性                                     │
│  [降级率趋势]  [熔断触发时间线]  [错误率]          │
└──────────────────────────────────────────────────┘

```

---

## 9、自部署 vs 云服务

| 方案  | 优点  | 缺点  | 推荐场景  |
|-------|-------|-------|-------------|
| LangFuse Cloud  | 零运维、即开即用  | 数据出境、月费  | 个人项目 / 早期验证  |
| LangFuse 自部署  | 数据不出境、无月费  | 需要维护 Postgres + 服务  | **企业生产环境 推荐**  |
| 替代方案 LangSmith  | LangChain 生态原生  | 闭源、数据出境  | 纯 LangChain 项目  |

Globex 推荐**自部署**：docker-compose 里加一个 LangFuse 服务即可：
```yaml
langfuse:
  image: langfuse/langfuse:2
  environment:
    - DATABASE_URL=postgresql://postgres:postgres@langfuse-db:5432/langfuse
    - NEXTAUTH_SECRET=your-secret
    - NEXTAUTH_URL=http://localhost:3000
  ports:
    - "3000:3000"
  depends_on:
    - langfuse-db

langfuse-db:
  image: postgres:16-alpine
  environment:
    - POSTGRES_PASSWORD=postgres
  volumes:
    - langfuse-data:/var/lib/postgresql/data

```

---

**本章小结：**

到这里，Globex 有了完整的线上可观测性体系：
1. **LangFuse 接入四步**：SDK → Trace → Span → Score，LangGraph callback handler 自动记录每次 LLM 调用。
2. **四类 Trace 信息**：工具调用链路 / Token 归因（含 cache_hit）/ AgentLoop 轮次时间线 / RaR Score。
3. **5 分钟 badcase SOP**：Score 筛选 → Trace 展开 → Span 定位 → 根因确认。
4. **工具 RT 告警**：滑动窗口 P99 超阈值自动推送钉钉 / Slack。
5. **和 Rubric 评测互补**：Rubric 告诉你"哪条不好"，LangFuse 告诉你"为什么不好"。
6. **监控看板**：延迟 / Token / 质量 / 成本 / 稳定性五类指标一屏掌握。

下一章「[Token 预算管理与模型路由降级](16-4 Token预算管理与模型路由降级.md)」会讲怎么在 LangFuse 监控的基础上，对单条请求做实时成本控制——token 快花完时自动切到便宜模型，而不是让一条请求无限消耗。
