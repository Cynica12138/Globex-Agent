# 16-2 vLLM 推理服务与 GPU 部署

## 本章课程目标

- 理解为什么 Agent 场景用 vLLM 而不是原生 transformers——PagedAttention、continuous batching、tool_call 原生解析三件事缺一不可。
- 掌握 vLLM 服务在 Docker 内的完整部署配置：GPU 挂载、模型加载、tool_call parser、healthcheck。
- 理解 GPU 利用率从 <20% 提升到 78% 的三板斧：batch_size / FP16 / continuous batching。
- 掌握 Reranker 独立 GPU 服务的必要性和部署方式。
- 拿到一份延迟预算表：从用户发 query 到最终 ShoppingSummary，每一步的 P50/P99 预算。

**学习建议：** 这一章聚焦“模型推理怎么快且稳”。第 16-1 章在 docker-compose 里给了 vLLM 和 Reranker 的 service 骨架，本章深入它们的内部配置——怎么让 35B MoE 模型在 A100 上跑得又快又省。

**对应代码分支：** `16-2-vllm-gpu-deployment`

---

## 1、为什么用 vLLM 而不是原生 transformers

### 1.1 原生 transformers 在 Agent 场景的三个问题

| 问题 | 具体表现 |
|---|---|
| 无 PagedAttention | KV Cache 固定分配，35B 模型一次只能服务 1-2 条并发请求 |
| 无 continuous batching | 一条请求结束前其他请求只能排队，GPU 大量空闲 |
| 无 tool_call 原生解析 | 模型输出 tool_call JSON 需要自己写 parser，格式错误率高 |

直接用 transformers + `model.generate()`，单卡 A100 的 GPU 利用率通常 <20%，同时只能服务 1-2 个并发 Agent 请求。

### 1.2 vLLM 解决了什么

| 特性 | 解决什么 |
|---|---|
| **PagedAttention** | KV Cache 按页动态分配，显存利用率提升 2-4 倍，并发从 2 → 8+ |
| **Continuous Batching** | 新请求随到随拼，不等上一条结束，GPU 几乎无空闲 |
| **Tool Call Parser** | 内置 Hermes / Llama3 格式解析，模型输出直接解析成结构化 tool_call |
| **OpenAI 兼容 API** | 直接暴露 `/v1/chat/completions`，LangChain `init_chat_model` 无缝对接 |

### 1.3 Globex 的收益

```text
transformers 直接推理：
  并发 2 / GPU 利用率 18% / 单请求 P99 ~8s

vLLM 服务：
  并发 8+ / GPU 利用率 78% / 单请求 P99 ~2.5s
```

---

## 2、vLLM 服务的完整部署配置

### 2.1 Docker 内启动 vLLM

16-1 章 `docker-compose.yml` 里的 vLLM service 块已经给了骨架，这里展开讲每个参数：

```yaml
vllm:
  image: vllm/vllm-openai:v0.8.5
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=0
  command: >
    --model Qwen/Qwen3-35B-A3B
    --served-model-name globex-main
    --tensor-parallel-size 1
    --max-model-len 16384
    --gpu-memory-utilization 0.90
    --enable-auto-tool-choice
    --tool-call-parser hermes
    --max-num-seqs 16
    --port 8000
```

### 2.2 关键参数详解

| 参数 | 推荐值 | 含义 |
|---|---|---|
| `--model` | Qwen/Qwen3-35B-A3B | HuggingFace 模型名或本地路径 |
| `--served-model-name` | globex-main | API 里的模型别名，Agent 代码里用这个名字 |
| `--tensor-parallel-size` | 1（单卡）/ 2（双卡） | MoE 模型单卡能放下就不切，切了通信开销大 |
| `--max-model-len` | 16384 | 和 SFT/RL 训练时的 max_seq_len 对齐 |
| `--gpu-memory-utilization` | 0.90 | 留 10% 给 CUDA 运行时，太高容易 OOM |
| `--enable-auto-tool-choice` | 必须开 | 让模型输出 tool_call 时自动切入解析模式 |
| `--tool-call-parser` | hermes | Qwen3 用 Hermes 格式；Llama3 用 llama3_json |
| `--max-num-seqs` | 16 | 最大并发序列数，决定 continuous batching 的上限 |
| `--port` | 8000 | 容器内端口，compose 里映射到宿主 8100 |

### 2.3 MoE 模型的特殊考量

Qwen3-35B-A3B 是 MoE（8 专家激活 3），总参数 35B 但每个 token 只走 ~14B 参数。

| 项目 | Dense 35B | MoE 35B（如 Qwen3-35B-A3B） |
|---|---|---|
| 显存占用 | 35B × 2 字节 = 70GB | 35B × 2 = 70GB（全部专家都要加载） |
| 推理速度 | 每 token 走 35B 参数 | 每 token 走 ~14B 参数 → 快 2.5 倍 |
| 单卡能否放下 | A100 80GB 勉强 | A100 80GB 轻松（90% 利用率有余量） |
| tensor-parallel | 2 卡起步 | 1 卡够用 |

**结论**：MoE 模型在 vLLM 上推理速度和显存效率都更优，Globex 选 MoE 是“训练侧效率 + 推理侧效率”的双赢。

---

## 3、Agent 调用 vLLM 的两种方式

### 3.1 方式一：OpenAI-compatible API（推荐）

vLLM 暴露和 OpenAI 完全兼容的 `/v1/chat/completions` 接口，LangChain 直接对接：

```python
# app/agent/llm.py（已有，这里展示 vLLM 对接部分）
from langchain.chat_models import init_chat_model

def get_llm():
    return init_chat_model(
        "globex-main",              # --served-model-name
        model_provider="openai",
        api_key="not-needed-local", # 本地 vLLM 不需要真 key
        base_url="http://vllm:8000/v1",  # Docker 内部网络
        temperature=0.3,
    )
```

**好处**：Agent 代码不需要知道后端是 vLLM 还是 OpenAI API——只换 `base_url` 就能切换。

### 3.2 方式二：vLLM Python Client（特殊场景）

如果需要更细粒度的控制（如流式 + tool_call 混合）：

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://vllm:8000/v1",
    api_key="not-needed",
)

response = await client.chat.completions.create(
    model="globex-main",
    messages=messages,
    tools=tool_schemas,
    tool_choice="auto",
    stream=True,
)
```

一般情况下用方式一（LangChain 封装）就够了，方式二留给需要手动控制流式 + tool_call 混合场景。

---

## 4、GPU 利用率优化三板斧

### 4.1 问题：单条推理 GPU 利用率 <20%

```text
Agent 请求进来 → 发一条给 vLLM → 等 vLLM 生成完 → 下一条才进
GPU 大部分时间在等 Agent 处理 Observation / 调工具 / 压缩上下文
```

GPU 的算力利用率取决于**同时在 GPU 上跑的序列数**。单条请求时 GPU 只有一个序列在跑，利用率自然低。

### 4.2 三板斧

#### 板斧 1：增大 max-num-seqs（并发序列上限）

```text
--max-num-seqs 16
```

允许 vLLM 同时处理 16 条请求。当 Agent A 在 Think 阶段生成 token 时，Agent B 的 Observe 结果也在 prefill——GPU 始终有活干。

| max-num-seqs | GPU 利用率 | 单请求 P99 | 吞吐量（req/s） |
|---|---:|---:|---:|
| 1 | 18% | 2.1s | 0.5 |
| 4 | 52% | 2.3s | 1.8 |
| 8 | 68% | 2.5s | 3.2 |
| **16** | **78%** | **2.8s** | **5.5** |
| 32 | 85% | 3.8s | 7.0 |

Globex 选 16：利用率 78% 且单请求延迟可控（P99 < 3s）。32 的延迟已经让用户体验下降。

#### 板斧 2：FP16 量化

vLLM 默认已经用 FP16（半精度），比 FP32 显存减半、推理速度提升约 40%。如果显存紧张还可以用 AWQ INT4，但精度损失明显——Agent 的 tool_call 格式准确率会掉 2-3%。

**Globex 选 FP16**——在 A100 80GB 上放 35B MoE 绰绰有余，不需要牺牲精度。

#### 板斧 3：Continuous Batching（vLLM 默认开启）

不需要额外配置——vLLM 天然支持。新请求到达时直接插入当前 batch，不等已有请求结束。

```text
传统 static batching：
  请求 A 跑完（3s）→ 请求 B 开始 → 请求 C 排队
  GPU 在 B 结束前不接 C

continuous batching：
  请求 A 在跑 → 请求 B 来了直接插入 → 请求 C 也来了也插入
  每个 decode step 所有活跃序列一起推理
```

### 4.3 优化后的整体效果

| 指标 | 优化前（transformers） | 优化后（vLLM + 三板斧） |
|---|---:|---:|
| GPU 利用率 | < 20% | 78% |
| 并发请求数 | 1-2 | 16 |
| 单请求 P50 | 5.2s | 1.8s |
| 单请求 P99 | 8.1s | 2.8s |
| 吞吐量 | 0.5 req/s | 5.5 req/s |

---

## 5、Reranker 独立 GPU 服务

### 5.1 为什么必须和 vLLM 分卡

Reranker（BGE-Reranker-v2-m3，567M 参数）虽然小，但它的推理模式和 LLM 完全不同：

| 维度 | LLM（vLLM） | Reranker |
|---|---|---|
| 推理模式 | autoregressive，逐 token 生成 | encode-only，一次性编码全部输入 |
| 显存模式 | KV Cache 动态增长 | 固定（batch_size × seq_len） |
| 峰值显存 | 波动大 | 稳定 |
| 调用频率 | 每轮 Think 调一次 | 只在 CategoryInsight 时调 |

共用同一张卡时，LLM 的 KV Cache 动态增长会挤压 Reranker 的固定显存空间，导致 Reranker 推理时触发 GPU OOM 或延迟飙升。

**结论**：LLM 用 GPU 0，Reranker 用 GPU 1（或独立的 T4/A10 小卡）。

### 5.2 Reranker 服务的 Dockerfile

```dockerfile
# docker/Dockerfile.reranker
FROM python:3.10-slim AS base

RUN pip install --no-cache-dir \
    torch==2.3.0 \
    transformers==4.44.0 \
    fastapi==0.115.0 \
    uvicorn==0.30.0

COPY app/recall/reranker_service.py /app/server.py

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.3 Reranker 服务代码骨架

```python
# app/recall/reranker_service.py
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()
MODEL_NAME = "BAAI/bge-reranker-v2-m3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval().half().cuda()  # FP16 + GPU


class RerankRequest(BaseModel):
    query: str
    candidates: list[str]


class RerankResponse(BaseModel):
    scores: list[float]


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    pairs = [[req.query, c] for c in req.candidates]
    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        ).to("cuda")
        scores = model(**inputs).logits.squeeze(-1).float().tolist()
    return RerankResponse(scores=scores)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

### 5.4 性能基线

| 配置 | Top-100 候选 P50 | Top-100 候选 P99 |
|---|---:|---:|
| CPU（单线程） | 480ms | 620ms |
| GPU FP32（A10） | 65ms | 95ms |
| **GPU FP16（A10）** | **35ms** | **52ms** |
| GPU FP16 + batch=128 | 28ms | 45ms |

Globex 用 FP16 + batch，P99 < 50ms。

---

## 6、延迟预算表：从用户发 query 到清单展示

### 6.1 完整链路延迟分解

一条“跨 4 平台搜旅行三件套”的典型链路：

| 阶段 | 组件 | P50 | P99 | 备注 |
|---|---|---:|---:|---|
| Think 1（Planner 决策） | vLLM | 800ms | 1.5s | 首次 Think 含 system prompt prefill |
| Act 1（Planner 工具） | CPU | 50ms | 100ms | 纯规则解析 |
| Think 2（fork 决策） | vLLM | 600ms | 1.2s | 上下文已 prefill 过，快 |
| Act 2（4 路 ItemSearch 并行） | 三塔+ANN | 1.2s | 2.0s | 瓶颈在最慢平台的 ANN 检索 |
| Think 3 | vLLM | 500ms | 1.0s | — |
| Act 3（PriceCompare） | CPU | 80ms | 150ms | 纯计算 |
| Act 4（ShippingCalc） | CPU | 30ms | 60ms | 查表 |
| Think 4 | vLLM | 500ms | 1.0s | — |
| Act 5（ItemPicker） | CPU | 60ms | 120ms | 规则+轻计算 |
| Think 5 | vLLM | 500ms | 1.0s | — |
| Act 6（ShoppingSummary） | vLLM | 1.5s | 2.5s | 生成最终 Markdown 文本 |
| **总计** | — | **5.8s** | **10.6s** | — |

### 6.2 延迟预算的工程意义

| 预算档位 | 阈值 | 触发什么 |
|---|---|---|
| 正常 | < 12s | 不做任何干预 |
| 告警 | 12-20s | LangFuse 告警 + 检查 vLLM 是否过载 |
| 降级 | > 20s | Token 预算触发强制压缩 + 模型路由降级 |
| 超时 | > 300s | 任务强制终止 + 给用户已有中间结果 |

### 6.3 哪些环节是延迟大头

```text
vLLM Think 阶段（5 次）：占总延迟 50-60%
  -> 优化方向：减少 Think 次数（更好的 system prompt 引导一步到位）

ItemSearch 并行召回：占总延迟 20-30%
  -> 优化方向：ANN 索引预热 + 更快的平台 API

ShoppingSummary 生成：占总延迟 15-20%
  -> 优化方向：限制生成长度 / 用流式推送让用户边看边等
```

---

## 7、冷启动预热

### 7.1 什么是冷启动

vLLM 服务刚启动时，第一批请求的延迟会比稳态高 3-5 倍：

- CUDA context 初始化
- KV Cache 首次分配
- 模型权重从 CPU → GPU 的首次搬运
- PagedAttention 的页表首次建立

### 7.2 预热策略

在 healthcheck 通过之后、接入真实流量之前，跑一批假请求把推理路径“暖起来”：

```python
# scripts/warmup_vllm.py
import httpx
import asyncio

WARMUP_PROMPTS = [
    {"role": "user", "content": "帮我搜旅行三件套"},
    {"role": "user", "content": "想买咖啡杯"},
    # ... 共 50-100 条
]

async def warmup():
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, msg in enumerate(WARMUP_PROMPTS):
            await client.post(
                "http://localhost:8100/v1/chat/completions",
                json={
                    "model": "globex-main",
                    "messages": [{"role": "system", "content": "你是 Globex"}, msg],
                    "max_tokens": 50,
                },
            )
            if i % 10 == 0:
                print(f"Warmup {i}/{len(WARMUP_PROMPTS)}")
    print("Warmup done.")

if __name__ == "__main__":
    asyncio.run(warmup())
```

在 `docker-compose` 里可以用一个 one-off service 来跑：

```yaml
warmup:
  build:
    context: ../
    dockerfile: docker/Dockerfile
    target: base
  command: python scripts/warmup_vllm.py
  depends_on:
    vllm:
      condition: service_healthy
  profiles: ["warmup"]
```

启动方式：

```bash
docker-compose --profile warmup run warmup
```

### 7.3 预热后的效果

| 指标 | 冷启动首批 10 条 | 预热后稳态 |
|---|---:|---:|
| P50 | 4.2s | 1.8s |
| P99 | 7.5s | 2.8s |
| GPU 利用率峰值 | 90%+（不稳定） | 78%（稳定） |

---

## 8、和其它章节的关系

| 章节 | 本章为它解决了什么 |
|---|---|
| 第 10 章 agent/llm.py | `get_llm()` 里的 `base_url` 指向 vLLM 服务 |
| 第 11 章 ItemSearch | 三塔编码走独立 Embedding 服务，和 vLLM 不争资源 |
| 第 13 章 Reranker | Reranker 独立 GPU 服务，本章给了完整部署 |
| 16-1 Docker Compose | vLLM service 块的每个参数在本章展开讲 |
| 16-3 可观测性 | vLLM 的延迟分位数是 LangFuse 监控的核心指标之一 |
| 16-4 Token 预算 | 模型路由降级时，可能从 vLLM 切到更小的模型实例 |

---

## 本章小结

到这里，Globex 的 LLM 推理和 Reranker 推理都有了生产级的服务化方案：

1. **vLLM 是 Agent 推理的标配**——PagedAttention + continuous batching + tool_call 解析，单卡 A100 从 2 并发提到 16 并发。
2. **GPU 利用率三板斧**：max-num-seqs=16 / FP16 / continuous batching，利用率从 <20% → 78%。
3. **MoE 模型天然适合 vLLM**——35B 总参但每 token 只走 14B，单卡轻松放下且推理快。
4. **Reranker 必须和 LLM 分卡**——两者显存模式不同，共卡会互相 OOM。
5. **冷启动预热**：服务 healthy 后跑 50-100 条假请求，首批请求延迟从 4.2s 降到 1.8s。
6. **延迟预算表**：完整链路 P50 ~5.8s / P99 ~10.6s，vLLM Think 占 50-60% 是优化大头。

下一章「[可观测性体系与 LangFuse 全链路 Trace](16-3 可观测性体系与LangFuse全链路Trace.md)」会讲怎么把 vLLM 的延迟、Token 消耗、工具调用链路全部接进 LangFuse，让线上 badcase 5 分钟内定位到根因。
