# 第 14 章：CategoryInsight RAG 工程化——从「最小可用」到「上线敢推」

## 本章课程目标

- 把第 13 章「最小可用」的 RAG 链路推到「上线敢推」的工程档位：补齐数据生产、Hybrid 调参、Rerank 精排、召回评测四个真正决定上限的环节。
- 学会用一份完整的 OpenSearch `search_pipeline` 配置做权重调参，并清楚什么时候把 BM25 权重降到接近 0。
- 在召回链路上增加 Cross-Encoder Rerank，把 Recall@8 从 0.6 档拉到 0.8 档。
- 搭一套最小召回评测脚手架（30～50 条标注集 + Recall@K / MRR / NDCG），让 CategoryInsight 每次升级都有客观刹车。

> **学习建议：**  
> 这一章是第 13 章的「工程化补丁」。如果你只想跑通 Demo，第 13 章已经够了；但如果你要把 CategoryInsight 真正接到线上、还要持续优化，下面四节才是命门。看代码时把第 13 章 §3 和 §5 打开对照——本章基本上是在那两节的位置上贴补丁。

---

## 1、本章导读

### 1.1 第 13 章为什么「轻」

第 13 章把链路跑通的代价，是把 RAG 的几个深水区「打了占位符」：

| 第 13 章的处理 | 真实工程里的隐患 |
|---|---|
| 卡片「假定已有」 | 卡片质量决定召回上限，没有数据生产管线就等于在空气上建房子 |
| `search_pipeline` 只给名字 | normalization / weights 没给，跑起来不知道权重是否合理 |
| 粗排出来直接进提炼 | Top-K 里夹了 1～2 张完全不相关的卡片，提炼出的 insight 就会跑偏 |
| 提炼后就交给主 loop | 升级一版召回不知道有没有变好——可能 case-by-case 看起来更好，整体却倒退 |

这四件事，对应本章四条主线。

### 1.2 本章节奏

```text
节 1 数据生产管线          → 让卡片「够用」
节 2 完整 Hybrid DSL 调参 → 让粗排「够准」
节 3 Rerank 精排（浅档）  → 让 Top-K「够干净」
节 4 召回评测脚手架       → 让每次升级「有刹车」
节 5 工程清单（收尾）     → 多语言 / 冷启动 / 兜底 / 成本，一句话各表
```

---

## 2、数据生产管线：让卡片本身够好

### 2.1 数据从哪来

Globex 知识库的三类卡片（爆款 / 属性图谱 / 价格区间）背后是四路原始数据：

| 数据源 | 喂给哪类卡片 | 频率 |
|---|---|---|
| 内部销售榜 | 爆款卡片 | 周 |
| 平台公开榜单 | 爆款卡片 | 周 |
| 商品库属性聚合 | 属性图谱卡片 | 月 |
| 历史成交价分位 | 价格区间卡片 | 月 |

四路数据在原始形态上完全不同（CSV / API / 数仓查询 / 平台爬虫），不能直接灌进 OpenSearch。中间需要一段离线 ETL，把它们统一收敛成 `CategoryCard` 这一种结构。

### 2.2 ETL 的三步

```text
原始数据
  → Step 1：标准化（字段统一、品类名归一）
  → Step 2：小模型抽取（生成 summary + raw_evidence）
  → Step 3：入库门禁（confidence 评分 + 抽审）
  → CategoryCard 写入 OpenSearch
```

### 2.3 Step 1：字段标准化

四路数据的第一个公分母，是**品类名归一**。

「旅行收纳」 / 「旅行三件套」 / 「便携收纳包」在不同数据源里的写法可能不同，但它们最终对应的卡片应该指向同一个标准品类。

```python
# app/recall/category_norm.py

CATEGORY_ALIASES: dict[str, str] = {
    "旅行收纳": "旅行三件套",
    "便携收纳包": "旅行三件套",
    "出差三件套": "旅行三件套",
    "咖啡杯": "咖啡杯",
    "马克杯": "咖啡杯",
    # ...离线靠人工 + 商品图谱维护，规模一般在几百条
}


def normalize_category(raw: str) -> str:
    raw = raw.strip().lower()
    return CATEGORY_ALIASES.get(raw, raw)
```

归一表本身不是「自动产生」的，而是由商品运营 + 数据团队共同维护的一张表。

**Globex 的工程边界是：把这张表当作 ground truth，不让 LLM 在线上临时猜品类映射。**

### 2.4 Step 2：小模型抽取生成 summary

`CategoryCard.summary` 的写法是有约定的。第 13 章 §5.3 的规则提炼，正是依赖这些约定格式才能稳定解析。

可以让小模型按指定格式生成：

```python
# scripts/etl/extract_card.py

from app.agent.llm import get_judge_llm   # 同 judge 共用强模型


EXTRACT_PROMPT = """
你是 Globex 品类知识库的卡片抽取器。

输入：一段关于品类 {category} 的原始资料（评测 / 销售榜 / 商品库聚合）。
输出：一张严格按格式写的 CategoryCard.summary。

约定格式（任选一种，根据 card_type）：
  bestseller:  "{{category}}：{{组件1}} / {{组件2}} / {{组件3}}"
  attribute:   "材质：尼龙 60% / 帆布 25% / 牛津布 15%"
  price_range: "便宜款 60-150 / 中档 150-400 / 高端 400+ 多见品牌联名"

raw_evidence 字段额外输出 1-3 条原始文本，每条不超过 80 字。
confidence: 0-1 之间，基于「原始数据量」和「措辞确定性」自评。

只输出 JSON，不要解释。
"""


async def extract_card(category: str, raw_text: str, card_type: str) -> dict:
    llm = get_judge_llm()

    resp = await llm.ainvoke([
        ("system", EXTRACT_PROMPT.format(category=category)),
        ("user", f"card_type={card_type}\n\n资料：\n{raw_text}"),
    ])

    import json
    return json.loads(resp.content)
```

抽取完成后，立刻按约定格式做一次校验。

格式错误的卡片直接 reject，不进入下一步。

> **核心原则：**  
> 让小模型为后续规则提炼负责，而不是让规则提炼去为小模型的「自由发挥」擦屁股。

### 2.5 Step 3：入库门禁

不是抽出来就能直接进库。

Globex 设计三道串行门禁：

```python
# scripts/etl/admit.py

from pydantic import ValidationError
from app.recall.category_kb import CategoryCard


MIN_CONFIDENCE = 0.5
MAX_SUMMARY_LEN = 200
SAMPLE_AUDIT_RATIO = 0.1   # 10% 的卡片走人工抽审


def admit(raw: dict) -> tuple[bool, str]:
    # 门 1：schema 严格校验
    try:
        card = CategoryCard(**raw)
    except ValidationError as e:
        return False, f"schema 校验失败: {e}"

    # 门 2：confidence + 长度
    if card.confidence < MIN_CONFIDENCE:
        return False, f"confidence {card.confidence} < {MIN_CONFIDENCE}"

    if len(card.summary) > MAX_SUMMARY_LEN:
        return False, f"summary 超长 {len(card.summary)}"

    # 门 3：summary 格式约定校验
    if card.card_type == "bestseller" and "：" not in card.summary:
        return False, "bestseller summary 缺少品类前缀"

    if card.card_type == "attribute" and "%" not in card.summary:
        return False, "attribute summary 缺少百分比"

    return True, "ok"
```

其中 10% 的卡片同时进入人工抽审队列。

抽审是**离线异步**进行的，不阻塞主 ETL 流程。

### 2.6 数据生产的产出

跑一轮全量 ETL，大致会得到：

| 阶段 | 数量级 |
|---|---:|
| 原始资料 | ~50,000 段 |
| 标准化通过 | ~32,000 段 |
| 抽取产出 | ~28,000 张草卡 |
| 门禁通过 | ~21,000 张 |
| 实际入库 | ~21,000 张 |
| 人工抽审 | ~2,100 张 |

后面第 5 节的召回评测，就可以从抽审结果中挑选高质量样本作为标注数据。

---

## 3、完整 Hybrid DSL 与权重调参

### 3.1 `search_pipeline` 完整配置

第 13 章 §5.2 的代码里有这么一行：

```python
params={"search_pipeline": "globex_hybrid_pipeline"},
```

那么 `globex_hybrid_pipeline` 到底长什么样？

可以一次性注册到 OpenSearch：

```bash
# 开发期可以放 scripts/setup_pipeline.sh

PUT _search/pipeline/globex_hybrid_pipeline
{
  "description": "KNN + BM25 双路召回的归一与加权融合",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {
          "technique": "min_max"
        },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [0.7, 0.3]
          }
        }
      }
    }
  ]
}
```

三个关键配置如下：

| 字段 | 含义 |
|---|---|
| `normalization=min_max` | KNN 余弦分（-1～1）和 BM25 分（0～30+）量纲完全不同，先各自归一到 `[0, 1]` |
| `combination=arithmetic_mean` | 算术平均，简单稳定，在小知识库上通常比 RRF 更容易控制 |
| `weights=[0.7, 0.3]` | KNN 权重 0.7，BM25 权重 0.3 |

`weights` 数组的顺序，对应第 13 章 §5.2 中：

```python
body["query"]["hybrid"]["queries"]
```

数组里的子查询顺序。

**子路顺序和权重顺序必须一一对应，否则很容易把 KNN / BM25 权重调反。**

### 3.2 权重调参的经验取值

`weights=[0.7, 0.3]` 不是拍脑袋拍出来的，而是根据品类 query 的形态分档调出来的：

| 品类 query 形态 | 推荐 weights `[KNN, BM25]` | 原因 |
|---|---|---|
| 名词为主，如「咖啡杯」「螺丝刀」 | `[0.5, 0.5]` | 字面匹配本身就很准，BM25 不弱 |
| 偏属性约束，如「防水旅行三件套」 | `[0.7, 0.3]` | KNN 支撑语义，BM25 兜字面长尾 |
| 偏气质形容，如「中性气质的咖啡杯」 | `[0.9, 0.1]` | 「中性气质」这类词 BM25 很难命中，主要依赖语义 |
| 完全口语，如「想送男朋友的礼物」 | `[1.0, 0.0]` | 可以关闭 BM25 子路，走纯 KNN |

Globex v1 默认：

```text
[KNN, BM25] = [0.7, 0.3]
```

因为它覆盖最常见的「品类 + 属性」查询形态。

如果后续要做按 query 形态自适应权重，可以在主 loop 的 Think 阶段判断 query 类型，再选择不同的 pipeline 或不同的召回分支。

### 3.3 什么时候关掉 BM25 子路

并不是所有场景都应该保留 BM25。

例如：

```text
中性气质的咖啡杯
```

这类完全语义化的 query，BM25 往往主要命中「咖啡杯」三个字，然后返回一堆字面相关、语义却不对的卡片。

结果就是：

```text
BM25 杂项结果
    ↓
参与 Hybrid 融合
    ↓
把 KNN 真正命中的「极简 / 性冷淡 / 中性设计」卡片挤出 Top-K
```

因此可以增加一个简单的判定型分支：

```python
# app/tools/category_insight.py
# 召回前预判

SEMANTIC_TOKENS = {
    "气质",
    "感觉",
    "风格",
    "感",
    "适合",
    "送",
    "氛围",
}


def should_disable_bm25(category: str) -> bool:
    """品类 query 含明显语义化 token 时，关闭 BM25 子路。"""
    return any(token in category for token in SEMANTIC_TOKENS)
```

然后在第 13 章 §5.2 的 `_build_hybrid_body()` 中，根据这个判定决定是否加入 BM25 查询子路。

这里的思想比「永远固定 `[0.7, 0.3]`」更重要：

> **Hybrid 不是为了永远同时跑两路，而是为了针对 query 特征选择最合适的召回信号。**

---

## 4、Rerank 精排：补上 RAG 的「最后一公里」

### 4.1 为什么粗排到这里还不够

第 13 章直接把 Hybrid 召回的 Top-8 喂给规则提炼。

但实践里，Top-8 经常会夹着 1～2 张「相关但跑题」的卡片。

例如查询：

```text
旅行三件套
```

召回结果里可能混进：

```text
旅行洗漱包
```

它命中了「旅行」，但它不是「三件套」。

如果这张卡片进入后续规则提炼，就可能让 `components` 字段出现错位。

因此更稳妥的方案是：

```text
Hybrid 粗排 Top-30
        ↓
Cross-Encoder Rerank
        ↓
精排 Top-8 / Top-15
        ↓
规则提炼
```

### 4.2 双塔与 Cross-Encoder 的区别

双塔模型的思路是：

```text
Query → Query Encoder → query vector
Item  → Item Encoder  → item vector

最后计算相似度
```

也就是：

> 你算你的，我算我的，最后再比较相似度。

优点是快，可以提前离线计算 Item Embedding，非常适合做大规模粗排。

Cross-Encoder 则是：

```text
[Query, Candidate]
        ↓
同一个模型联合编码
        ↓
直接输出相关性分数
```

Query 和 Candidate 会发生 token 级交互，因此相关性判断会细很多。

代价也很明显：**慢得多**。

所以两类模型不是二选一，而是分工：

```text
双塔 / KNN        → 粗排，解决「从海量候选里快速捞出来」
Cross-Encoder     → 精排，解决「这几十个里谁才是真的相关」
```

### 4.3 模型选型

| 候选 | 优点 | 缺点 | 推荐场景 |
|---|---|---|---|
| `BGE-Reranker-v2-m3` | 开源 / 多语言 / 性能稳定 | 需要本地或自建推理服务 | Globex 默认 ✅ |
| `Cohere Rerank v3` | API 即用 / 多语言能力强 | 付费 / 可能涉及数据出境 | 个人项目 / 早期 Demo |
| LLM-as-Reranker | 灵活、可解释 | 慢 + 贵 | 课程外延伸，本章不展开 |

本章采用：

```text
BGE-Reranker-v2-m3 + 本地服务
```

它支持中英文等多语言场景，比较适合 Globex 的跨境购物 Query。

### 4.4 Reranker 客户端

```python
# app/recall/reranker.py

import os
import httpx
from typing import Sequence


class RerankerClient:
    """BGE-Reranker-v2-m3 的极简客户端。

    要求服务端暴露 /rerank：

    入参：
      {
        "query": str,
        "candidates": list[str]
      }

    出参：
      {
        "scores": list[float]
      }

    scores 与 candidates 保持同序。
    """

    def __init__(self) -> None:
        self.endpoint = os.environ["RERANKER_ENDPOINT"]
        self._client = httpx.AsyncClient(timeout=3.0)

    async def score(
        self,
        query: str,
        candidates: Sequence[str],
    ) -> list[float]:
        response = await self._client.post(
            self.endpoint,
            json={
                "query": query,
                "candidates": list(candidates),
            },
        )

        response.raise_for_status()
        return response.json()["scores"]


reranker = RerankerClient()
```

### 4.5 接入 `_recall_cards`

第 13 章 §5.2 的 `_recall_cards()` 内部原来直接取 Top-K=8。

现在改成：

```text
Hybrid 粗排 30
    ↓
Rerank
    ↓
Quick：保留 8
Deep：保留 15
```

代码如下：

```python
# app/tools/category_insight.py
# 替换原 _recall_cards

COARSE_K = 30
FINE_K_QUICK = 8
FINE_K_DEEP = 15

RERANK_BYPASS_TOP_SCORE = 0.92


async def _recall_cards(
    category: str,
    top_k: int,
) -> list[CategoryCard]:
    # 1. Query 向量
    emb = await tower_client.encode_query(category)

    # 2. Hybrid 粗排 Top-30
    body = _build_hybrid_body(
        category,
        emb,
        coarse_k=COARSE_K,
    )

    resp = _kb_client.search(
        index=INDEX_NAME,
        body=body,
        params={
            "search_pipeline": "globex_hybrid_pipeline",
        },
    )

    hits = resp["hits"]["hits"]

    if not hits:
        return []

    # 3. 短路 1：
    # 粗排首位分数足够高，说明当前粗排已经非常确定，
    # 直接跳过 rerank，节省一次推理延迟。
    if hits[0]["_score"] >= RERANK_BYPASS_TOP_SCORE:
        return [
            CategoryCard(**hit["_source"])
            for hit in hits[:top_k]
        ]

    # 4. 短路 2：
    # 候选本来就不超过 top_k，没有精排裁剪压力。
    if len(hits) <= top_k:
        return [
            CategoryCard(**hit["_source"])
            for hit in hits
        ]

    # 5. Cross-Encoder 精排
    candidates_text = [
        hit["_source"]["summary"]
        for hit in hits
    ]

    scores = await reranker.score(
        category,
        candidates_text,
    )

    paired = sorted(
        zip(scores, hits),
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        CategoryCard(**hit["_source"])
        for _, hit in paired[:top_k]
    ]
```

这里有两个非常重要的「短路」：

| 短路条件 | 意义 |
|---|---|
| 粗排首位置分高于阈值 | 粗排已经非常确定，Rerank 增量很小 |
| 粗排候选 `<= Top-K` | 候选本身就少，没有排序裁剪压力 |

所以：

> **Rerank 不是无脑加，而是按需要才走。**

### 4.6 加入 Rerank 后的实测变化

| 指标 | 仅 Hybrid 粗排 | 粗排 + Rerank | 备注 |
|---|---:|---:|---|
| Recall@8 | ~0.62 | ~0.81 | 主要收益来源 |
| MRR | ~0.55 | ~0.74 | 第一条卡片更「对」 |
| 单次延迟 P50 | ~25 ms | ~75 ms | Rerank 一次约 +50 ms |
| 单次延迟 P99 | ~80 ms | ~140 ms | 短路命中率约 30% |

P50 增加约 50 ms，在整个 Agent RAG 子链路中通常可以接受。

主 AgentLoop 的一轮 Think 本身就需要数百毫秒，因此这部分延迟换来的相关性提升是值得的。

---

## 5、召回评测脚手架：让升级有刹车

### 5.1 评测应该放在哪一层

Globex 已经有第 8 章的 Rubric 端到端评测。

但**召回评测不是替代 Rubric，而是补它的盲区。**

| 评测类型 | 关心什么 | 成本 / 标注方式 |
|---|---|---|
| Rubric（第 8 章） | 端到端商品清单质量 | 慢、贵、依赖 Judge LLM |
| 召回评测（本节） | CategoryInsight 单点召回质量 | 快、便宜、纯结构化指标 |

如果 CategoryInsight 只是：

- 改了一行召回代码；
- 调整一次 weights；
- 改了 `coarse_k`；
- 换了一版 Reranker；

这些修改不应该每次都跑完整 Rubric。

> **召回评测就是模块级的「日常体检」。**

### 5.2 标注集长什么样

v1 阶段不需要几千条。

Globex 可以先准备：

```text
50 条典型品类 Query

覆盖：
- 名词类
- 属性类
- 气质类
- 口语类

每条 Query：
由商品运营人工挑选 5 张「应该被召回」的 CategoryCard，
并按重要性排序。

最终：
50 × 5 = 250 个 (query, card_id) ground truth 对。
```

存储格式：

```json
{"query": "旅行三件套", "relevant": ["c_001", "c_017", "c_042", "c_088", "c_101"]}
{"query": "中性气质的咖啡杯", "relevant": ["c_220", "c_233", "c_251", "c_268", "c_271"]}
```

文件路径：

```text
data/eval/category_recall.jsonl
```

每行一条，方便后续增量补标。

### 5.3 三个核心指标

```python
# app/eval/recall_metrics.py

from typing import Sequence


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
) -> float:
    """Top-K 召回结果覆盖了多少标注相关卡片。"""
    top_k = set(retrieved[:k])
    rel = set(relevant)

    if not rel:
        return 0.0

    return len(top_k & rel) / len(rel)


def mrr(
    retrieved: Sequence[str],
    relevant: Sequence[str],
) -> float:
    """首条相关卡片的倒数排名。"""
    rel = set(relevant)

    for i, rid in enumerate(retrieved, start=1):
        if rid in rel:
            return 1.0 / i

    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k: int,
) -> float:
    """NDCG@K：同时考虑位置与标注顺序。"""
    import math

    # 越靠前的 relevant card，gain 越高
    rel_rank = {
        rid: len(relevant) - i
        for i, rid in enumerate(relevant)
    }

    dcg = sum(
        rel_rank.get(rid, 0) / math.log2(i + 2)
        for i, rid in enumerate(retrieved[:k])
    )

    ideal = sum(
        rel_rank[rid] / math.log2(i + 2)
        for i, rid in enumerate(relevant[:k])
    )

    return dcg / ideal if ideal else 0.0
```

三者分别关注：

| 指标 | 关心什么 | 什么时候最重要 |
|---|---|---|
| Recall@K | 标注的卡片有没有被找回来 | 任何召回环节的底线 |
| MRR | 第一条相关结果出现得有多早 | Top-1 会直接影响下游 ItemPicker 时 |
| NDCG@K | 排序质量 | 不只看有没有命中，还看高质量卡片是否靠前 |

可以简单理解为：

```text
Recall@K：找没找回来？
MRR：第一张对的卡片排第几？
NDCG@K：整体排序是不是把更重要的卡片放前面？
```

### 5.4 跑测脚本

```python
# scripts/eval/run_category_recall.py

import asyncio
import json
from pathlib import Path

from app.tools.category_insight import _recall_cards
from app.eval.recall_metrics import (
    recall_at_k,
    mrr,
    ndcg_at_k,
)


EVAL_PATH = Path(
    "data/eval/category_recall.jsonl"
)

TOP_K = 10


async def main() -> None:
    samples = [
        json.loads(line)
        for line in EVAL_PATH.open(
            encoding="utf-8"
        )
    ]

    recall_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0

    for sample in samples:
        cards = await _recall_cards(
            sample["query"],
            top_k=TOP_K,
        )

        retrieved = [
            card.card_id
            for card in cards
        ]

        recall_sum += recall_at_k(
            retrieved,
            sample["relevant"],
            TOP_K,
        )

        mrr_sum += mrr(
            retrieved,
            sample["relevant"],
        )

        ndcg_sum += ndcg_at_k(
            retrieved,
            sample["relevant"],
            TOP_K,
        )

    n = len(samples)

    print(
        f"Recall@{TOP_K} = "
        f"{recall_sum / n:.3f}"
    )

    print(
        f"MRR          = "
        f"{mrr_sum / n:.3f}"
    )

    print(
        f"NDCG@{TOP_K}   = "
        f"{ndcg_sum / n:.3f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
```

50 条标注全量跑一次大约几秒量级，完全适合常驻 CI。

### 5.5 回归门禁

把这套指标接到发版流程：

| 门禁指标 | Globex v1 阈值 | 触发后处理 |
|---|---:|---|
| Recall@10 | ≥ 0.75 | 低于阈值，阻断发版 |
| MRR | ≥ 0.65 | 低于阈值，阻断发版 |
| NDCG@10 | ≥ 0.70 | 低于阈值，告警 + 人工评审是否发版 |

任何对以下参数或组件的修改：

```text
weights
coarse_k
rerank model
rerank threshold
BM25 disable rule
embedding model
```

都应该先跑一次召回评测，过门禁后再合代码。

---

## 6、收尾工程清单

下面这些内容不再展开成独立大节，但真正上线时都要有。

### 6.1 多语言对齐

跨境购物天然涉及多语种 Query。

常见有两条工程路线：

| 路径 | 优点 | 缺点 | 推荐 |
|---|---|---|---|
| Query 翻译归一到中文索引 | 索引便宜 / 维护轻 | 翻译失真 / 长尾词可能丢失 | Globex v1 ✅ |
| 多语言并行建索引 | 召回更准 / 长尾损失更小 | 索引膨胀 N 倍 / 维护成本高 | 业务规模扩大后再上 |

翻译归一可以放在 `_recall_cards()` 入口前。

例如：

```text
English Query
    ↓
Query 归一 / 多语言 Embedding
    ↓
统一语义空间
    ↓
Hybrid Recall
```

如果 Query Tower 本身已经支持多语言统一向量空间，也可以避免显式翻译。

### 6.2 冷启动：新品类无卡片

对于刚上线的新类目，例如：

```text
户外烧水壶
```

知识库可能还没有卡片。

这时 `_recall_cards()` 会拿不到任何 hit。

退化链路：

```text
1. Recall 命中 0
      ↓
2. CategoryInsight 返回空 insight
      ↓
3. 主 loop 判断：
   components / bestsellers 全空
      ↓
4. 调 WebSearch 兜底
      ↓
5. WebSearch 结果由小模型抽成草卡
      ↓
6. 草卡仅供本轮使用，不在线写入知识库
```

关键边界：

> **在线工具只负责使用临时结果，不负责污染正式知识库。**

正式入库依旧只能由离线 ETL + 入库门禁负责。

### 6.3 Query 级缓存

同一个 category 在几小时内可能被反复查询。

例如用户在一个会话里连续多次提到：

```text
旅行三件套
```

没必要每次都重新走：

```text
Embedding
  + Hybrid
  + Rerank
  + Insight 提炼
```

可以在 Redis 加一层轻量缓存：

```python
# 伪代码

cache_key = (
    f"cinsight:{category}:{depth}"
)

cached = await redis.get(cache_key)

if cached:
    return CategoryInsightOutput.model_validate_json(
        cached
    )

# ...正常召回与提炼...

await redis.setex(
    cache_key,
    3600,
    result.model_dump_json(),
)
```

TTL 设成：

```text
3600 秒
```

对于「卡片周更 / 月更」的业务节奏已经足够保守。

### 6.4 异常 / 空召回兜底

| 异常类型 | 兜底策略 |
|---|---|
| OpenSearch 不可用 | 返回 `confidence=0` 的空 insight + 上报 |
| Reranker 超时 | 跳过精排，直接使用粗排 Top-K |
| Tower 不可用 | 退化到纯 BM25 查询 |
| 召回为空 | 走 WebSearch 冷启动兜底 |

所有这些兜底都尽量：

```text
不直接向主 loop 抛异常
```

而是统一返回：

```text
低置信度、结构化、可继续决策的结果
```

这样主 AgentLoop 才能在下一轮 Think 中继续判断：

```text
是调用 WebSearch？
还是降级？
还是 ChatFallback 跟用户对齐？
```

### 6.5 工程指标看板

真正上线后，至少要监控以下指标：

| 指标 | Globex v1 期望值 |
|---|---:|
| Recall@10 | ≥ 0.75 |
| 单次 P50 延迟 | ≤ 80 ms |
| 单次 P99 延迟 | ≤ 200 ms |
| Rerank 短路命中率 | ≥ 25% |
| 空召回率 | ≤ 2% |
| Cache 命中率 | ≥ 30% |

任意一项跌出正常波动区间，例如：

```text
1.5σ
```

就触发告警。

---

## 7、本章小结

到这里，Globex 的 RAG 链路就从：

```text
跑得通
```

升级到了：

```text
上线敢推
```

完整工程链路可以概括为：

```text
                离线数据生产
原始数据
    ↓
标准化
    ↓
小模型抽取
    ↓
入库门禁
    ↓
CategoryCard
    ↓
OpenSearch
    │
    ├───────────────┐
    │               │
   KNN             BM25
    │               │
    └──── Hybrid ───┘
             ↓
         粗排 Top-30
             ↓
      Cross-Encoder Rerank
             ↓
         精排 Top-K
             ↓
        规则 / LLM 提炼
             ↓
      CategoryInsight
             ↓
        主 AgentLoop
```

本章四个最核心的升级点如下。

### 7.1 数据生产管线

```text
标准化
  → 小模型抽取
  → 入库门禁
```

目的是让卡片本身成为高质量结构化数据，而不是一堆随意文本。

### 7.2 完整 Hybrid DSL

```text
KNN + BM25
    ↓
Min-Max 归一
    ↓
加权融合
```

Globex v1 默认：

```text
[KNN, BM25] = [0.7, 0.3]
```

再根据 Query 形态进行动态调整，必要时直接关闭 BM25 子路。

### 7.3 Rerank 精排

```text
Hybrid Top-30
    ↓
BGE-Reranker-v2-m3
    ↓
Top-8 / Top-15
```

同时增加两条短路：

```text
粗排首位置信度很高 → 跳过 Rerank
候选数 <= Top-K      → 跳过 Rerank
```

把 Recall@8 从约：

```text
0.62
```

推到约：

```text
0.81
```

代价是 P50 增加约 50 ms。

### 7.4 召回评测脚手架

```text
50 条人工标注
    +
Recall@K
    +
MRR
    +
NDCG
    +
CI 回归门禁
```

让 CategoryInsight 的每次升级都有客观刹车。

### 7.5 收尾工程能力

最终还要补齐：

```text
多语言归一
冷启动 WebSearch
Redis Cache
异常降级
空召回兜底
工程监控看板
```

读完这一章后再回到第 13 章看 `_recall_cards()`，就能明显看到：

> 第 13 章给的是一个「能跑的 RAG 脚手架」，而这一章补齐的，才是一个真正接近线上工程形态的 CategoryInsight RAG 链路。

---

## 下一章

**主 AgentLoop 组装与同质子 AgentLoop Fork 协同机制**
