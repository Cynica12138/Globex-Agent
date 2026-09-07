# 08-1 Agent SFT 冷启动训练全流程

## 本章课程目标

- 理解 Agent 轨迹和通用 Q-A 在训练数据层面的本质差异——多轮、长序列、工具返回不可控。
- 掌握 Agent SFT 训练数据的三档来源 + 冷启动期的造数据策略 + 数据清洗五条红线。
- 理解 Agent SFT 特有的 Loss Mask 策略——为什么工具返回不该算 loss、怎么落到代码上。
- 拿到一份可直接复用的训练配方：Curriculum Learning 三阶段 + 关键超参 + 长序列工程。
- 掌握 SFT 阶段的评测标准：什么时候才算“训好了、可以交给 RL”。

## 学习建议

这一章是 [第 8 章 Rubric 评测与 Agentic RL 训练闭环](08%20Rubric评测与Agentic-RL训练闭环.md) 的 **SFT 深度补完**——第 8 章讲了“为什么需要 SFT”和“数据从哪来”的概念，本章讲“怎么把数据造出来、loss 怎么设计、训练怎么跑、怎么知道训好了”。

读时对照三条主线：

1. **Agent 轨迹 ≠ 通用 Q-A**：多轮结构 + 工具返回 + 长度不固定，训练方法必须适配。
2. **Loss Mask 是 Agent SFT 的命门**：不 mask 工具返回，模型会学到“生成工具输出”的错误目标。
3. **格式正确率 ≥ 98% + Rubric 分 ≥ 65 = 可以交给 RL**：这是 SFT → RL 的交接标准。

---

## 1、Agent 轨迹 vs 通用 Q-A 的本质区别

### 1.1 通用 SFT：一问一答

通用大模型的 SFT 数据长这样：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "帮我翻译：旅行背包"},
    {"role": "assistant", "content": "Travel backpack"}
  ]
}
```

结构简单、长度短，**模型只需要学会生成 assistant 那一段**。

### 1.2 Agent SFT：多轮 Think → Act → Observe → Reflect 的完整轨迹

Globex 的一条 SFT 训练样本长这样（简化展示）：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是 Globex 购物 Agent...（完整 system prompt）"
    },
    {
      "role": "user",
      "content": "想买便宜又抗造的旅行三件套，预算 300，不要塑料"
    },

    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "tc_001",
          "type": "function",
          "function": {
            "name": "planner",
            "arguments": "{\"query\": \"旅行三件套 预算300 不要塑料\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "tc_001",
      "content": "{\"budget\": 300, \"category\": \"旅行三件套\", ...}"
    },

    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "tc_002",
          "type": "function",
          "function": {
            "name": "dispatch_tool",
            "arguments": "{\"demands\": \"在 amazon 搜旅行三件套...\"}"
          }
        },
        {
          "id": "tc_003",
          "type": "function",
          "function": {
            "name": "dispatch_tool",
            "arguments": "{\"demands\": \"在 shopee 搜旅行三件套...\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "tc_002",
      "content": "[{\"item_id\": \"A01\", \"title\": \"...\"}, ...]"
    },
    {
      "role": "tool",
      "tool_call_id": "tc_003",
      "content": "[{\"item_id\": \"S01\", \"title\": \"...\"}, ...]"
    },

    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "tc_004",
          "type": "function",
          "function": {
            "name": "price_compare",
            "arguments": "..."
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "tc_004",
      "content": "{\"ranked\": [...]}"
    },

    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "tc_005",
          "type": "function",
          "function": {
            "name": "item_picker",
            "arguments": "..."
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "tc_005",
      "content": "{\"picks\": [...]}"
    },

    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "tc_006",
          "type": "function",
          "function": {
            "name": "shopping_summary",
            "arguments": "..."
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "tc_006",
      "content": "{\"final_text\": \"## 推荐 3 件\\n...\"}"
    },

    {
      "role": "assistant",
      "content": "## 推荐 3 件\n1. **NORDIC TRAVEL SET** ..."
    }
  ]
}
```

### 1.3 三个本质差异

| 维度 | 通用 Q-A | Agent 轨迹 |
|---|---|---|
| 结构 | 1 轮 user → 1 轮 assistant | N 轮 assistant + tool 交替 |
| 长度 | 通常 < 2K token | 通常 8K-16K token（含工具返回） |
| 模型应学会什么 | 生成 assistant 内容 | 生成 Think + tool_call JSON，**不是工具返回** |
| 数据构造难度 | 低（标注 / 改写即可） | 高（工具返回不确定、分支路径多、格式必须严格） |

---

## 2、训练数据怎么造

### 2.1 三档来源加权采样

| 档位 | 来源 | 质量 | 规模 | 推荐采样权重 |
|---|---|---:|---:|---:|
| 一档 | RaR 高分轨迹自动入库（线上） | 最高 | 持续增长 | **50%** |
| 二档 | 强模型蒸馏（GPT-5 / Qwen-Max） | 高 | ~500 条 | **30%** |
| 三档 | 人工构造示范（运营手工跑） | 中-高 | 50-100 条 | **20%** |

**关键点：**

- 一档是飞轮跑起来后的主力。但 Day 0 没有线上数据时，一档为空——冷启动全靠二档 + 三档。
- 二档的核心操作：让强模型在 Globex 的工具环境里跑同一条 query，把完整轨迹录下来当教师信号。
- 三档的核心操作：商品运营用“人工操作 Agent”模式（human-in-the-loop）手动选择每一步调哪个工具、传什么参数。

### 2.2 冷启动期造数据策略

```text
Day 0：线上数据 = 0
   ↓
Step 1：运营团队手动跑 50 条标杆 query（三档）
Step 2：用强模型跑同样 50 条 query + 额外 450 条新 query（二档）
Step 3：合并 → 第一版 SFT 数据集（~550 条轨迹）
Step 4：用第一版数据集训出 SFT v0
Step 5：v0 上线 → RaR 评测系统自动跑 → 高分轨迹入库（一档开始积累）
Step 6：每周合并新的一档数据 → 训 SFT v1, v2, ...
```

550 条听起来少，但 Agent 轨迹的信息密度远高于通用 Q-A——每条平均 10 轮 Think + Act，等效于 5000+ 个决策点。

### 2.3 强模型蒸馏的具体操作

```python
# scripts/distill_trajectories.py
import asyncio

from app.agent.main_agent import run_agent
from app.agent.llm import get_judge_llm
from app.eval.rubric import evaluate_trajectory


DISTILL_QUERIES = [
    "想买便宜又抗造的旅行三件套，预算 300，不要塑料",
    "帮我找中性气质的咖啡杯，送男朋友生日",
    # ... 500 条
]


async def distill_one(query: str) -> dict | None:
    """用强模型跑一条 query，评分合格则入库。"""

    # 1. 用强模型（而非业务模型）跑完整 Agent 轨迹
    result = await run_agent(
        query=query,
        thread_id=f"distill-{hash(query)}",
        model_override=get_judge_llm(),  # 关键：用强模型
    )

    # 2. RaR 评测打分
    score = await evaluate_trajectory(result["trajectory"])
    if score < 70:
        return None  # 强模型也可能跑出低分轨迹，直接丢弃

    # 3. 格式标准化后入库
    return standardize_trajectory(result["trajectory"])
```

### 2.4 轨迹格式标准化

不管数据来自哪一档，入库前必须对齐同一个格式：

| 字段 | 要求 |
|---|---|
| `tool_calls[].function.name` | 必须是 `FULL_TOOL_SET` 中的合法工具名 |
| `tool_calls[].function.arguments` | 必须是可 `JSON.parse` 的字符串 |
| `tool_calls[].id` | 必须唯一，格式 `tc_NNN` |
| `<think>` 标签 | assistant 的推理过程必须包在 `<think>...</think>` 中 |
| `role: "tool"` 消息 | 必须有对应的 `tool_call_id` |

格式不对齐的轨迹在训练时会引入噪声——模型学到“有时候 tool_call 没有 id、有时候有”，推理时就会概率性省略 id。

### 2.5 数据清洗五条红线

每条轨迹入库前必须过自动校验：

| 红线 | 阈值 | 触发后处理 |
|---|---:|---|
| 无效循环 | 同工具连续 ≥ 4 次 | 整条轨迹剔除 |
| 过长轨迹 | 总 token > 16K | 截断到最后一次完整工具调用 |
| 格式错误 | tool_call JSON 不可 parse | 整条轨迹剔除 |
| 结果泄露 | 输出含 `item_id` 或内部工具名 | 整条轨迹剔除 |
| Rubric 分数不达标 | < 70 分 | 不入库 |

```python
# app/eval/trajectory_admit.py
import json


def admit_trajectory(traj: list[dict]) -> tuple[bool, str]:
    """五条红线校验。"""

    # 红线 1：无效循环
    tools_seq = [
        m["tool_calls"][0]["function"]["name"]
        for m in traj
        if m.get("tool_calls")
    ]

    for i in range(len(tools_seq) - 3):
        if len(set(tools_seq[i:i + 4])) == 1:
            return False, f"无效循环: {tools_seq[i]} 连续 4 次"

    # 红线 2：过长
    total_tokens = sum(len(m.get("content", "")) // 3 for m in traj)
    if total_tokens > 16000:
        return False, f"过长: ~{total_tokens} token"

    # 红线 3：格式错误
    for m in traj:
        for tc in m.get("tool_calls", []):
            try:
                json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                return False, "tool_call arguments 不可 parse"

    # 红线 4：泄露
    full_text = " ".join(
        m.get("content", "")
        for m in traj
        if m["role"] == "assistant"
    )

    if "item_id" in full_text or "dispatch_tool" in full_text:
        return False, "输出泄露内部字段"

    return True, "ok"
```

> 实际工程里还需要把 **Rubric < 70** 的判断接入 admission pipeline；上面的示例函数主要展示轨迹结构层面的自动校验。

---

## 3、Loss 设计与 Mask 策略

### 3.1 问题：标准 Causal LM Loss 不适配 Agent 轨迹

标准做法是对整条序列每一个 token 计算 cross-entropy loss。

但 Agent 轨迹里有大量 `role: "tool"` 消息——它们是**工具返回的外部数据**，模型不应该被训练去“生成”它们。

如果不 mask：

```text
模型会学到：
  - “在 assistant 之后，应该生成一段 JSON 格式的工具返回数据”
  - 这不是模型应该做的事——工具返回是外部环境给的
  - 训出来的模型可能会“自己编造”工具返回，而不是等待真正的工具执行
```

### 3.2 三档 Mask 策略对比

| 策略 | 在哪些 token 上算 loss | 效果 | 推荐 |
|---|---|---|---|
| A：全算（baseline） | 所有 token | 差，模型会学习工具返回 | ❌ |
| B：只在 assistant 上算 | `role: "assistant"` 的全部 content 和 tool_calls | 好 | **推荐** |
| C：进一步细分 | Think + tool_call 全权，Reflect 降 0.5 权 | 稍好，收益有限 | 进阶可选 |

### 3.3 Mask 构建的核心代码

```python
# scripts/train/build_loss_mask.py
from typing import Sequence


def build_agent_loss_mask(
    messages: Sequence[dict],
    tokenizer,
    max_length: int = 16384,
) -> tuple[list[int], list[int]]:
    """构建 Agent SFT 的 input_ids + loss_mask。

    loss_mask[i] = 1 表示该 token 参与 loss 计算；
    loss_mask[i] = 0 表示该 token 被 mask。
    """

    input_ids: list[int] = []
    loss_mask: list[int] = []

    for msg in messages:
        role = msg["role"]

        # 编码当前消息的全部 token
        # 实际实现需要结合具体模型的 chat_template，
        # 这里仅做逻辑示意。
        content = msg.get("content", "")
        tool_calls_str = ""

        if msg.get("tool_calls"):
            import json
            tool_calls_str = json.dumps(
                msg["tool_calls"],
                ensure_ascii=False,
            )

        text = f"{content}{tool_calls_str}"
        tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        input_ids.extend(tokens)

        if role == "assistant":
            # assistant 的 token 全部参与 loss
            loss_mask.extend([1] * len(tokens))
        else:
            # system / user / tool 的 token 全部 mask
            loss_mask.extend([0] * len(tokens))

    # 截断到 max_length
    input_ids = input_ids[:max_length]
    loss_mask = loss_mask[:max_length]

    return input_ids, loss_mask
```

**核心逻辑只有一条：**

```text
role == "assistant" → loss_mask = 1
其它 role             → loss_mask = 0
```

也就是只训练模型真正需要“生成”的部分。

### 3.4 Mask 的工程影响

| 不 mask（策略 A） | 正确 mask（策略 B） |
|---|---|
| 约 60%-70% 的 loss 花在工具返回上 | loss 集中在模型真正需要学习的 token 上 |
| 模型容易学“编造工具结果” | 模型学习“什么时候调什么工具 + 参数怎么生成” |
| 格式正确率可能较低 | 格式正确率可显著提升 |
| 收敛慢，loss 被无关 token 稀释 | 收敛更快，决策信号更集中 |

---

## 4、训练配方与稳定性

### 4.1 Curriculum Learning 三阶段

| 阶段 | 时间 | 数据 | 目标 | 关键超参调整 |
|---|---|---|---|---|
| Phase 1 | 1-2 天 | 短轨迹（≤ 4 轮 Act） | 学格式 + 基础调用 | `lr=2e-5`, `warmup=200` |
| Phase 2 | 2-3 天 | 中等轨迹（4-8 轮 Act） | 学调用顺序 + fork | `lr=1e-5`, `warmup=100` |
| Phase 3 | 3-5 天 | 全量轨迹 + 复杂 fork 场景 | 学决策质量 | `lr=5e-6`, `warmup=50` |

为什么要 Curriculum：

```text
直接用全量数据训
    ↓
模型一开始见大量长轨迹
    ↓
gradient 被长序列主导
    ↓
短轨迹里的格式信号被淹没
    ↓
格式正确率迟迟上不去

先短后长
    ↓
模型先把格式学会
    ↓
再学长链路里的调用顺序与决策逻辑
    ↓
训练更稳定
```

### 4.2 关键超参表

| 超参 | 推荐值 | 原因 |
|---|---|---|
| 学习率（Phase 1/2/3） | `2e-5 / 1e-5 / 5e-6` | 逐阶段递减，降低灾难遗忘风险 |
| Batch size（有效） | 32-64 条轨迹 | Agent 轨迹长，单条 token 数已经很大 |
| Gradient accumulation | 8-16 步 | 配合小 micro-batch 凑到目标有效 batch |
| Max sequence length | 16384 token | 覆盖约 95% 的轨迹 |
| Warmup steps | `200 / 100 / 50` | 与各阶段学习率配合 |
| Weight decay | 0.01 | 常用稳定配置 |
| Epoch | 3-5（按 Phase） | 结合 eval 指标决定是否提前停止 |

### 4.3 长序列问题与工程解法

Agent 轨迹平均 8K-16K token，一条样本就会占用大量显存。

| 问题 | 解法 | 效果 |
|---|---|---|
| 显存爆炸 | Gradient Checkpointing | 重计算换显存，显存可显著下降 |
| 单卡放不下 | DeepSpeed ZeRO-3 | 参数 / 梯度 / 优化器状态跨卡切片 |
| 短轨迹 padding 浪费 | Sequence Packing | 多条短轨迹拼成一条，提高 token 利用率 |

Sequence Packing 在 Phase 1 最明显：例如把 4-5 条约 2K token 的短轨迹拼成一条接近 10K token 的训练序列，减少 padding 浪费。

### 4.4 过拟合的信号与应对

**信号：**

- Train loss 持续平稳下降；
- 但 eval 集上的格式正确率反而从 98% 降到 95%；
- 或 eval 集 Rubric 分不涨反跌。

这说明模型可能在“背”训练集里的特定轨迹模式，而不是学习可泛化的决策逻辑。

**应对：**

| 手段 | 说明 |
|---|---|
| Early Stopping | 取 eval 指标最好的 checkpoint，而不是 train loss 最低的 |
| 增大数据多样性 | 加更多强模型蒸馏数据 / query 变体 |
| Dropout | 在 LoRA adapter 层加 0.05-0.1 dropout |
| 降低 lr | Phase 3 出现过拟合时，可继续将 lr 减半 |

---

## 5、SFT 阶段评测——怎么知道训好了

### 5.1 离线模块级指标

每个 epoch 结束后，用一套固定 eval query 集合让 SFT 模型跑完整 Agent 轨迹，统计以下指标：

| 指标 | 计算方式 | 目标 |
|---|---|---:|
| 格式正确率 | tool_call JSON 可 parse 且字段完整的 case 占比 | **≥ 98%** |
| 工具调用成功率 | 调用正确工具 + 参数合法 | **≥ 90%** |
| 终结工具使用率 | 最终调用 ShoppingSummary 或 ChatFallback 正常收尾 | **≥ 95%** |
| 无效循环率 | 同工具连续 ≥ 4 次的 case 占比 | **< 3%** |
| 平均轮次 | 从 user query 到终结工具的 Act 轮数 | **5-8 轮** |

```python
# scripts/eval/eval_sft_format.py
async def eval_format_metrics(model, eval_queries: list[str]) -> dict:
    """跑一遍 eval 集，统计格式指标。"""

    import json

    results = {
        "format_ok": 0,
        "tool_ok": 0,
        "terminal_ok": 0,
        "loop": 0,
    }

    total = len(eval_queries)

    for q in eval_queries:
        traj = await run_agent_with_model(model, q)
        messages = traj["messages"]

        # 1. 格式正确率
        all_format_ok = True

        for m in messages:
            for tc in m.get("tool_calls", []):
                try:
                    json.loads(tc["function"]["arguments"])
                except Exception:
                    all_format_ok = False

        if all_format_ok:
            results["format_ok"] += 1

        # 2. 终结工具使用率
        last_tool = None

        for m in reversed(messages):
            if m.get("tool_calls"):
                last_tool = m["tool_calls"][-1]["function"]["name"]
                break

        if last_tool in {"shopping_summary", "chat_fallback"}:
            results["terminal_ok"] += 1

        # 3. 无效循环
        tools_seq = [
            m["tool_calls"][0]["function"]["name"]
            for m in messages
            if m.get("tool_calls")
        ]

        has_loop = any(
            len(set(tools_seq[i:i + 4])) == 1
            for i in range(len(tools_seq) - 3)
        )

        if has_loop:
            results["loop"] += 1

    return {
        "format_correct_rate": results["format_ok"] / total,
        "terminal_rate": results["terminal_ok"] / total,
        "loop_rate": results["loop"] / total,
    }
```

> 示例代码主要演示格式正确率、终结率和无效循环率。实际评测时还应补上 `tool_ok` 的工具选择正确性与参数 schema 校验逻辑。

### 5.2 端到端 Rubric 评测

模块级指标只能回答：

> “模型会不会正确地使用 Agent 格式？”

但不能回答：

> “它做出来的整条任务轨迹到底好不好？”

所以关键 checkpoint 还需要复用第 8 章的完整 Rubric 评测流程：

```text
动态生成 Rubric
    ↓
SFT Agent 执行完整任务
    ↓
Judge 按 Rubric 打分
    ↓
统计 100-200 条 query 的平均分
```

**SFT 阶段目标：Rubric 均分 ≥ 65 / 100。**

这个阈值不需要特别高，因为 SFT 主要负责：

- 学会 Agent 范式；
- 学会合法工具调用；
- 学会基础工具顺序；
- 建立一个可供 RL 继续优化的稳定 policy。

从 65 推到 75+，才是 RL 阶段的主要任务。

### 5.3 评测节奏

| 时间点 | 跑什么 | 决策 |
|---|---|---|
| 每 epoch 结束 | 离线格式指标 | 决定是否 early stop |
| Phase 切换前 | 离线格式指标 + Rubric | 决定是否进入下一阶段 |
| 训练结束 | 全量 Rubric | 决定交给 RL 的 checkpoint |

---

## 6、与 08-2 Agentic RL 的衔接

### 6.1 SFT checkpoint 选哪个给 RL

双条件同时满足：

```text
格式正确率 ≥ 98%
AND
Rubric 均分 ≥ 65
```

如果多个 checkpoint 都满足，优先选择 **Rubric 分最高的 checkpoint**。

原因是 RL 需要一个已经具有稳定 Agent 行为的初始 policy。如果初始化模型连工具调用格式都不稳定，reward 会非常稀疏，训练过程也更容易失控。

### 6.2 SFT 后模型的能力边界

| 能力 | SFT 后的状态 |
|---|---|
| 格式正确 | ✅ 稳定 ≥ 98% |
| 工具选择正确 | ✅ 大部分场景能选对工具 |
| fork 判断准确 | ⚠️ 能 fork，但时机不总是最优 |
| 决策质量高 | ⚠️ Rubric 通常约 65-70 |
| 避免 reward hacking | ❌ 尚未真正学习 reward 约束 |

SFT 的本质是模仿：

```text
给定状态
    ↓
模仿高质量轨迹里的下一步动作
```

它解决的是：

> “一个正确的 Agent 轨迹应该长什么样？”

但它并没有真正回答：

> “多个都能完成任务的动作里，哪个能获得更高 reward？”

因此 SFT 会逐渐出现性能天花板。

想继续提高，就需要在真实 Agent rollout 上直接优化 reward，这正是下一章 [08-2 Agentic RL 训练全流程](08-2%20Agentic-RL训练全流程.md) 要解决的问题。

---

## 本章小结

到这里，你应该能完整理解 Globex Agent SFT 冷启动训练的全流程：

1. **Agent 轨迹 ≠ 通用 Q-A**  
   Agent 数据是多轮、长序列的 Think → Act → Observe → Reflect 轨迹，而且工具返回属于环境信息，不是模型应该生成的目标。

2. **三档训练数据来源**  
   - RaR 高分轨迹：飞轮跑起来后的主力；
   - 强模型蒸馏：冷启动期主要数据来源；
   - 人工示范：保证最初一批轨迹质量。

3. **冷启动可以从约 550 条高质量轨迹开始**  
   虽然轨迹数量不大，但每条包含多个 Agent 决策点，训练信号密度远高于普通 Q-A。

4. **Loss Mask 是 Agent SFT 的核心**  
   只在 `role="assistant"` 的 token 上计算 loss，system / user / tool token 全部 mask，避免模型学习“生成工具返回”。

5. **Curriculum Learning 先短后长**  
   Phase 1 学格式，Phase 2 学调用顺序与 fork，Phase 3 学复杂长链路决策。

6. **长序列训练需要专门工程优化**  
   Gradient Checkpointing + ZeRO-3 + Sequence Packing，解决 8K-16K Agent 轨迹带来的显存和训练效率问题。

7. **SFT → RL 的交接标准**  
   ```text
   格式正确率 ≥ 98%
   +
   Rubric 均分 ≥ 65
   =
   可以进入 Agentic RL
   ```

下一章 **「08-2 Agentic RL 训练全流程」** 将从这个 SFT checkpoint 出发，通过 rollout、Rubric Reward、GSPO 等方法继续优化 Agent 的真实决策质量，把 Rubric 分从约 65 推向 75+。
