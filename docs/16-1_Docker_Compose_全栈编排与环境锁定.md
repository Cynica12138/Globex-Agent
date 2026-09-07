# 16-1 Docker Compose 全栈编排与环境锁定

**本章课程目标：**
- 理解为什么 Agent 项目比普通 Web 服务更需要容器化——不只是"方便部署"，是"评测分可复现"的前提。
- 掌握 docker-compose.yml 的完整骨架：FastAPI + OpenSearch + Redis + Reranker GPU + Embedding 服务 + vLLM。
- 理解 multi-stage Dockerfile 的 dev / prod 参数隔离——同一份代码在不同环境跑出不同行为，且行为可控。
- 拿到一份本地首次启动的 SOP 和常见排障清单。

**学习建议：** 这一章是"从 demo 到生产"的第一步。你可能已经在第 15 章把 Globex 跑起来了，但跑起来靠的是本机 Python 环境 + 手动安装的 OpenSearch + 自己装的 Redis——换一台机器就全不行了。本章解决这个问题。

---

## 1、为什么 Agent 项目比普通 Web 更需要容器化

### 1.1 普通 Web 服务跨环境的问题

普通 Web 只有一个 Python 进程 + 一个数据库。跨环境出问题通常是"依赖版本不对"——装个正确版本就好了。

### 1.2 Agent 项目跨环境的问题要严重得多

Globex 的运行环境包含至少 6 个组件：

| 组件  | 环境差异会导致什么  |
|-------|----------------------------|
| Python + LangChain  | 版本差异导致 tool_call JSON 格式不同，格式正确率漂移  |
| CUDA + GPU 驱动  | Reranker FP16 推理行为不同，精排结果不一致  |
| OpenSearch  | 本地单节点 vs 服务器多分片，召回结果排序不同  |
| Redis  | 版本差异导致 Store 序列化不兼容  |
| Agent 运行参数  | fork 深度 / loop 上限 / 超时时间 不一致  |
| asyncio 行为  | Mac 和 Linux 的事件循环默认策略不同  |

最致命的后果：**评测分跨环境不可复现**——本机跑 Rubric 评测得 79 分，换一台机器跑同一批 query 得 50 分，你不知道是代码改坏了还是环境不对。

### 1.3 容器化解决什么

把 6 个组件 + 全部运行参数 + 全部版本号 锁进一份 docker-compose.yml

任何机器 docker-compose up → 得到完全一样的环境

评测分可复现 = 代码改动的效果可信

---

## 2、docker-compose.yml 完整骨架

### 2.1 服务拓扑
```text
┌─────────────────────────────────────────────────┐
│  docker-compose.yml                              │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ FastAPI  │  │ OpenSearch│  │    Redis     │  │
│  │ (Agent)  │  │ (向量+全文)│  │ (Store+Cache)│  │
│  └────┬─────┘  └──────────┘  └──────────────┘  │
│       │                                          │
│  ┌────┴──────────────────────────────────────┐  │
│  │           内部网络 (globex-net)             │  │
│  └────┬──────────────┬───────────────────────┘  │
│       │              │                           │
│  ┌────┴─────┐  ┌─────┴─────┐                   │
│  │  vLLM    │  │ Reranker  │                   │
│  │ (LLM推理)│  │ (精排GPU) │                   │
│  └──────────┘  └───────────┘                   │
└─────────────────────────────────────────────────┘

```

### 2.2 完整 docker-compose.yml
```yaml
# docker/docker-compose.yml
version: "3.9"

x-common-env: &common-env
  TZ: Asia/Shanghai
  PYTHONUNBUFFERED: "1"

services:
  # ======== 基础设施 ========
  opensearch:
    image: opensearchproject/opensearch:2.15.0
    environment:
      - discovery.type=single-node
      - plugins.security.disabled=true
      - OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "9200:9200"
    volumes:
      - opensearch-data:/usr/share/opensearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:9200/_cluster/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  # ======== GPU 推理服务 ========
  vllm:
    image: vllm/vllm-openai:v0.8.5
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - VLLM_MODEL=${LLM_MAIN:-Qwen/Qwen3-35B-A3B}
    command: >
      --model ${LLM_MAIN:-Qwen/Qwen3-35B-A3B}
      --served-model-name globex-main
      --tensor-parallel-size 1
      --max-model-len 16384
      --enable-auto-tool-choice
      --tool-call-parser hermes
      --port 8000
    ports:
      - "8100:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/health || exit 1"]
      interval: 15s
      timeout: 10s
      retries: 20
      start_period: 120s

  reranker:
    image: ghcr.io/globex/reranker-service:latest
    build:
      context: ../
      dockerfile: docker/Dockerfile.reranker
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
      - MODEL_NAME=BAAI/bge-reranker-v2-m3
    ports:
      - "8200:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 15
      start_period: 60s

  # ======== Agent 主服务 ========
  agent:
    build:
      context: ../
      dockerfile: docker/Dockerfile
      target: ${BUILD_TARGET:-prod}
    environment:
      <<: *common-env
      OPENAI_BASE_URL: http://vllm:8000/v1
      OPENAI_API_KEY: "not-needed-local"
      LLM_MAIN: globex-main
      RERANKER_ENDPOINT: http://reranker:8000/rerank
      OPENSEARCH_HOST: opensearch
      STORE_REDIS_URL: redis://redis:6379/2
    ports:
      - "8000:8000"
    depends_on:
      opensearch:
        condition: service_healthy
      redis:
        condition: service_healthy
      vllm:
        condition: service_healthy
      reranker:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  opensearch-data:
  redis-data:

networks:
  default:
    name: globex-net

```

### 2.3 关键设计点

| 设计点  | 原因  |
|----------|-------|
| `depends_on: condition: service_healthy`  | Agent 必须等 vLLM 模型加载完才能接流量  |
| vLLM `start_period: 120s`  | 35B 模型加载需要 60-90s，不给足时间会被判为不健康  |
| Reranker 和 vLLM 分卡  | 显存争抢会让 Reranker 推理延迟从 50ms 飙到 300ms+  |
| Agent 的 `BUILD_TARGET` 变量  | dev/prod 用同一份 compose 文件，切 target 切参数  |
| 内部网络 `globex-net`  | 服务间用服务名通信（如 `http://vllm:8000/`）  |

---

## 3、multi-stage Dockerfile：dev / prod 参数隔离

### 3.1 为什么要隔离

| 参数  | dev 值  | prod 值  | 不隔离的后果  |
|-------|--------|---------|-------------------|
| AGENT_LOOP_MAX_ITERATIONS  | 100  | 30  | dev 调试时需要放宽，prod 放宽=浪费  |
| AGENT_TIMEOUT_SEC  | 600  | 300  | dev 长超时方便断点，prod 必须短  |
| FORK_MAX_DEPTH  | 5  | 2  | dev 测试递归 fork，prod 必须限制  |
| LOG_LEVEL  | DEBUG  | WARNING  | DEBUG 日志在 prod 性能拖累明显  |
| COMPRESS_KEEP_RECENT  | 5  | 3  | dev 保留多一些方便 debug  |

### 3.2 完整 Dockerfile
```dockerfile
# docker/Dockerfile

# ========== base 层：公共依赖 ==========
FROM python:3.10-slim AS base

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 源码
COPY app/ ./app/
COPY data/ ./data/

# ========== dev 层：放宽参数 + DEBUG ==========
FROM base AS dev

ENV AGENT_LOOP_MAX_ITERATIONS=100 \
    AGENT_TIMEOUT_SEC=600 \
    FORK_MAX_DEPTH=5 \
    LOG_LEVEL=DEBUG \
    COMPRESS_KEEP_RECENT=5

EXPOSE 8000
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ========== prod 层：收紧参数 + WARNING ==========
FROM base AS prod

ENV AGENT_LOOP_MAX_ITERATIONS=30 \
    AGENT_TIMEOUT_SEC=300 \
    FORK_MAX_DEPTH=2 \
    LOG_LEVEL=WARNING \
    COMPRESS_KEEP_RECENT=3

EXPOSE 8000
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

```

### 3.3 切换方式
```bash
# 本地开发
BUILD_TARGET=dev docker-compose up --build

# 生产部署 / 评测跑批
BUILD_TARGET=prod docker-compose up --build

```

同一份 docker-compose.yml，只改 `BUILD_TARGET` 环境变量就能切换 dev/prod 行为。**评测跑批必须用 prod target**——这样评测分才和线上行为一致。

---

## 4、环境变量注入：.env 文件 vs Docker secrets

### 4.1 两种方式的适用场景

| 方式  | 适用  | 不适用  |
|-------|-------|----------|
| `.env` 文件  | 非敏感配置（端口 / 模型名 / 阈值）  | API Key / 数据库密码  |
| Docker secrets  | 敏感配置（API Key / 密码）  | 需要在 compose file 里引用的值  |

### 4.2 .env 文件
```dotenv
# docker/.env
BUILD_TARGET=prod
LLM_MAIN=Qwen/Qwen3-35B-A3B
OPENSEARCH_HOST=opensearch
STORE_REDIS_URL=redis://redis:6379/2
RERANKER_ENDPOINT=http://reranker:8000/rerank

```

### 4.3 敏感信息用 secrets（生产环境）
```yaml
# docker-compose.prod.yml（生产覆盖文件）
services:
  agent:
    secrets:
      - openai_api_key
      - tavily_api_key
    environment:
      OPENAI_API_KEY_FILE: /run/secrets/openai_api_key

secrets:
  openai_api_key:
    external: true
  tavily_api_key:
    external: true

```

本地开发直接 `.env` 写明文，生产用 secrets——**永远不要把 API Key 写进 Dockerfile 或 docker-compose.yml 里**。

---

## 5、healthcheck + 启动顺序编排

### 5.1 为什么 depends_on 不够

`depends_on` 默认只保证"容器已启动"，不保证"服务已就绪"。vLLM 容器启动后模型加载还要 90s，这期间 Agent 发请求过来全是 503。

`condition: service_healthy` 才是真正保证"服务已就绪才启动依赖方"的方式。

### 5.2 各服务 healthcheck 设计

| 服务  | 检查方式  | interval  | start_period  | 原因  |
|-------|-------------|---------|-------------|-------|
| OpenSearch  | `curl /_cluster/health`  | 10s  | 0s  | 启动快，不需要 start_period  |
| Redis  | `redis-cli ping`  | 5s  | 0s  | 秒级启动  |
| vLLM  | `curl /health`  | 15s  | 120s  | 模型加载 60-90s  |
| Reranker  | `curl /health`  | 10s  | 60s  | 模型加载 30-40s  |
| Agent  | `curl /health`  | 10s  | 0s  | 纯 Python，秒级启动  |

### 5.3 启动顺序
```text
OpenSearch + Redis（并行启动，无依赖）
  ↓ healthy
vLLM + Reranker（并行启动，无互相依赖）
  ↓ healthy
Agent（等所有上游 healthy 才启动）

```

---

## 6、本地首次启动 SOP
```bash
# Step 1: 进入 docker 目录
cd globex-agent/docker

# Step 2: 复制环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key（如果有外部服务要调的话）

# Step 3: 构建并启动（首次需要下载镜像，约 5-10 分钟）
BUILD_TARGET=dev docker-compose up --build

# Step 4: 等待所有服务 healthy（观察日志）
# 看到 "agent_1 | INFO: Uvicorn running on http://0.0.0.0:8000" 就绑定了

# Step 5: 验证
curl http://localhost:8000/health
# 应该返回 {"status": "ok"}

# Step 6: 跑一条 query 验证端到端
curl -X POST http://localhost:8000/api/task \
  -H "Content-Type: application/json" \
  -d '{"query": "帮我搜旅行三件套", "thread_id": "test-001"}'

```

### 6.1 常见排障

| 现象  | 原因  | 解法  |
|-------|-------|-------|
| vLLM 一直 unhealthy  | GPU 显存不够加载 35B 模型  | 换小模型或增加 GPU  |
| Agent 启动报 ConnectionRefused  | depends_on 没配 condition  | 确认 `condition: service_healthy`  |
| OpenSearch 启动报 max_map_count  | Linux 内核参数不够  | `sysctl -w vm.max_map_count=262144`  |
| Reranker 推理报 CUDA OOM  | 和 vLLM 共用同一张卡  | 确认 `NVIDIA_VISIBLE_DEVICES` 分卡  |
| Redis 数据丢失  | 没挂载 volume  | 确认 `volumes: - redis-data:/data`  |

---

---

## 7、和其它章节的关系

| 章节 | 本章为它解决了什么 |
|---|---|
| 第 8 章 Rubric 评测 | 评测跑批必须在 prod target 下跑，确保分数可复现 |
| 第 10 章 .env 配置 | docker-compose 的 environment 覆盖了 .env 里的值 |
| 第 14 章 防失控 | FORK_MAX_DEPTH 等参数由 Dockerfile target 锁定 |
| 第 15 章 FastAPI | Agent 服务变成了 compose 里的一个 service |
| 16-2 vLLM | vLLM 服务在 compose 里定义，本章给骨架，16-2 讲深 |

---

## 本章小结

到这里，Globex 从“本机手动安装”升级到了“一键容器化”：

1. `docker-compose.yml` 编排 6 个服务：Agent + OpenSearch + Redis + vLLM + Reranker + Embedding，内部网络互通。
2. multi-stage Dockerfile 用 dev/prod 两个 target 隔离运行参数——同一份代码，`BUILD_TARGET=prod` 就是生产行为。
3. healthcheck + `condition: service_healthy` 保证启动顺序：上游 ready 才启动下游。
4. 敏感信息用 secrets，不进 `.env` / Dockerfile。
5. 本地首次启动：`docker-compose up --build` 一条命令，15 分钟内全部就绪。

下一章 **16-2 vLLM 推理服务与 GPU 部署** 会深入 vLLM 的配置调优——PagedAttention、continuous batching、tool_call 支持、GPU 利用率从 `<20%` 到 `78%` 的三板斧。
