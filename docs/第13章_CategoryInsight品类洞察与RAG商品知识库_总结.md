# 第13章 CategoryInsight 品类洞察工具与 RAG 商品知识库总结

## 一、CategoryInsight 是什么

`CategoryInsight` 不是商品搜索工具，而是 Globex 链路里的**前置认知层**。

它解决的问题是：

> 在真正搜索、比价、精挑商品之前，先让 Agent 知道“这个品类通常长什么样、有哪些典型属性、价格大概怎么分档”。

它主要给两个后续环节提供知识：

- 给 `ItemSearch`：帮助拆分更合理的 sub-query。
- 给 `ItemPicker`：帮助判断候选商品是否符合品类常识。

例如用户说：

```text
想买一套旅行三件套，预算 300，不要塑料
```

CategoryInsight 可以先补充：

```text
典型组件：洗漱包 / 鞋包 / 数码线收纳
主流材质：尼龙 / 帆布 / 牛津布
价格区间：60-150 / 150-400 / 400+
```

这样后续搜索和筛选就不是“只看商品表面字段”，而是带着品类知识做决策。

---

## 二、CategoryInsight 在链路中的位置

它可以出现在两个位置。

### 1. ItemSearch 之前

适合用户需求本身依赖品类知识的场景。

例如：

```text
旅行三件套
```

Planner 虽然识别出了品类，但还不知道“三件套”具体应该包含什么。

此时可以先：

```text
CategoryInsight
    ↓
获得典型组件
    ↓
ItemSearch 拆多个 sub-query
```

### 2. ItemPicker 之前

适合已经有候选商品，但需要“懂行地精挑”。

例如：

```text
这 12 件商品里，哪几件更适合送礼？
```

此时流程可以是：

```text
ItemSearch
    ↓
PriceCompare / ShippingCalc
    ↓
CategoryInsight
    ↓
ItemPicker
```

CategoryInsight 给 ItemPicker 补充品类属性、主流价格档位和典型商品特征。

---

## 三、RAG 商品知识库放什么

Globex 不把完整测评文章或博主长文直接塞进知识库，而是提前整理成**结构化卡片**。

主要分三类。

| 卡片类型 | 典型内容 | 用途 |
|---|---|---|
| bestseller | 爆款、典型组件 | 帮助 ItemSearch 拆查询 |
| attribute | 材质、容量、防水等属性分布 | 帮助 ItemPicker 判断是否主流 |
| price_range | 便宜 / 中档 / 高端价格区间 | 帮助 ItemPicker 判断价格是否合理 |

核心思想是：

> 知识库里存“已经提炼过的品类知识”，而不是无限堆原始文本。

---

## 四、CategoryCard 数据结构

知识库中的基础对象是：

```python
class CategoryCard(BaseModel):
    card_id: str
    category: str
    card_type: Literal[
        "bestseller",
        "attribute",
        "price_range"
    ]
    summary: str
    raw_evidence: list[str]
    last_updated: str
    confidence: float
```

各字段含义：

- `category`：标准品类名。
- `card_type`：卡片类型。
- `summary`：已经提炼好的结论。
- `raw_evidence`：支撑结论的原始证据。
- `confidence`：数据可信度。
- `last_updated`：知识更新时间。

虽然知识库内部保留 `raw_evidence`，但**CategoryInsight 最终不会把这些原文返回给主 Agent**。

---

## 五、为什么使用 OpenSearch

知识库使用 OpenSearch，把以下内容存到同一条文档中：

```text
结构化字段
+
全文字段
+
Embedding 向量
```

核心字段包括：

```text
category
summary
raw_evidence
confidence
content_vector
```

`content_vector` 使用：

```text
knn_vector
+
HNSW
+
Faiss engine
+
cosine similarity
```

相比原来的：

```text
Faiss 索引
+
meta.json
```

这种方案不再需要维护：

```text
向量索引 id
    ↓
meta.json
    ↓
原始文档
```

OpenSearch 查询命中后：

```python
hit["_source"]
```

就能直接拿到完整 `CategoryCard`。

---

## 六、CategoryInsight 的核心输出

工具定义大致为：

```python
@tool
async def category_insight(
    category: str,
    depth: Literal["quick", "deep"] = "quick"
) -> CategoryInsightOutput:
    ...
```

输出结构：

```python
class CategoryInsightOutput(BaseModel):
    category: str
    components: list[str]
    bestsellers: list[Bestseller]
    attributes: list[AttributeDist]
    price_tiers: list[PriceTier]
    confidence: float
```

关键字段：

| 字段 | 含义 |
|---|---|
| components | 套装类商品通常包含哪些组件 |
| bestsellers | 典型爆款 |
| attributes | 品类主流属性分布 |
| price_tiers | 价格档位 |
| confidence | 整体置信度 |

---

## 七、quick 和 deep 的区别

### quick

```text
Top-K = 8
```

主要获取：

```text
components
bestsellers
price_tiers
```

不额外提取：

```text
attributes
```

适合低延迟场景。

### deep

```text
Top-K = 15
```

除了 quick 的内容，还会进一步提取：

```text
attributes
```

因此成本更高，但给 ItemPicker 的判断依据更完整。

---

# 八、RAG 的核心不是“向量检索”

这一章最重要的理解是：

> RAG ≠ 单纯 Vector Search。

完整过程应该是：

```text
召回
 ↓
提炼
 ↓
摘要
```

即：

```text
Recall
→ Extract
→ Summarize
```

最终给 Agent 的不是：

```text
5～15 篇原始文档
```

而是：

```text
结构化结论
```

这可以显著降低主 Agent 的上下文占用。

---

# 九、第一步：召回

召回使用：

```text
KNN 向量检索
+
BM25 全文检索
```

组成 Hybrid Search。

整体过程：

```text
category
   ↓
Query Tower
   ↓
embedding
   ↓
OpenSearch
   ├── KNN
   └── BM25
   ↓
Hybrid Fusion
   ↓
Top-K CategoryCard
```

---

## 十、为什么要 KNN + BM25

### KNN

解决：

```text
语义相似
```

例如：

```text
便携旅行收纳套装
```

即使卡片写的是：

```text
旅行三件套
```

向量仍可能召回。

### BM25

解决：

```text
关键词完全匹配
```

例如：

```text
帆布旅行背包
```

如果知识库中刚好存在：

```text
category = 帆布旅行背包
```

BM25 往往比纯语义检索更可靠。

所以：

```text
KNN 负责语义召回
BM25 负责关键词精确匹配
```

两者互补。

---

# 十一、Hybrid Search

OpenSearch 中同时执行两路：

```python
{
    "hybrid": {
        "queries": [
            {"knn": {...}},
            {"multi_match": {...}}
        ]
    }
}
```

其中：

```text
KNN 权重：0.7
BM25 权重：0.3
```

通过 search pipeline 做：

```text
score normalization
+
score combination
```

最终形成统一排序。

---

# 十二、第二步：提炼

召回得到的卡片会先按照：

```python
card_type
```

分组：

```text
bestseller
attribute
price_range
```

然后分别提取。

---

## 1. components

从 bestseller 卡片中提取套装组件。

例如：

```text
旅行三件套：洗漱包 / 鞋包 / 数码线收纳
```

得到：

```python
[
    "洗漱包",
    "鞋包",
    "数码线收纳"
]
```

---

## 2. bestsellers

从原始证据：

```text
name | price | reason
```

中提取：

```python
Bestseller(
    name=...,
    typical_price_cny=...,
    why_popular=...
)
```

例如：

```text
多功能洗漱包 | 89 | 干湿分离
```

变成：

```text
name = 多功能洗漱包
price = 89
why_popular = 干湿分离
```

---

## 3. attributes

例如知识库中：

```text
材质：尼龙 60% / 帆布 25% / 牛津布 15%
```

提取成：

```python
AttributeDist(
    name="材质",
    distribution={
        "尼龙": 0.6,
        "帆布": 0.25,
        "牛津布": 0.15
    }
)
```

---

## 4. price_tiers

例如：

```text
便宜款 60-150
中档 150-400
高端 400+
```

转换为：

```python
PriceTier(
    tier="budget",
    range_cny=(60, 150)
)
```

以及：

```text
mid
premium
```

等价格层级。

---

# 十三、第三步：摘要

提炼完成后，把多个卡片合并成统一：

```python
CategoryInsightOutput
```

例如：

```text
旅行三件套

典型组件：
洗漱包 / 鞋包 / 数码线收纳

主流价格：
60-150
150-400
400+

典型爆款：
多功能洗漱包
便携鞋包

confidence：
0.78
```

此时原始 RAG 文档已经被压缩。

---

# 十四、为什么最终不返回 raw_evidence

CategoryInsight 的设计原则是：

```text
工具内部看证据
主 Agent 只看结论
```

如果把召回的 5～15 张卡片全部塞给主 Agent，会导致：

```text
上下文变长
+
token 增加
+
主 loop 被无关细节污染
+
推理复杂度增加
```

因此：

```text
RAG 原文
      ↓
CategoryInsight 内部处理
      ↓
结构化 Insight
      ↓
主 Agent
```

---

# 十五、为什么 CategoryInsight 适合 fork

Globex 判断是否 fork 看三件事：

```text
1. 能否并行
2. 上下文是否需要隔离
3. 调用链是否 ≥ 3
```

CategoryInsight 的情况：

| 判断 | 情况 |
|---|---|
| 能并行 | 否 |
| 上下文隔离 | 有价值 |
| 调用链 ≥ 3 | 是 |

其典型调用链：

```text
Step 1
Query Tower Embedding

Step 2
OpenSearch Hybrid Recall

Step 3
结构化提炼 / 小模型抽取

Step 4
摘要合并
```

如果真实系统中的提炼阶段使用小模型：

```text
调用链长度自然 ≥ 3
```

---

# 十六、fork 的核心价值

如果主 Agent 自己完成所有流程，它可能看到：

```text
15 张 CategoryCard
+
raw_evidence
+
提炼中间结果
```

而 fork 后：

```text
主 Agent
   ↓
dispatch_tool
   ↓
子 Agent
   ├── embedding
   ├── hybrid search
   ├── rerank / 提炼
   └── summary
   ↓
CategoryInsightOutput
   ↓
主 Agent
```

主 Agent 最终只看到：

```text
组件
+
属性
+
价格
+
置信度
```

所以 fork 的价值并不是“为了 fork 而 fork”，而是：

```text
隔离中间上下文
+
封装复杂调用链
+
减少主 loop token
```

---

# 十七、quick 与 deep 的 fork 策略

一种合理工程折衷：

```text
quick
→ 主 Agent 直接调用 category_insight
```

因为查询较轻。

而：

```text
deep
→ dispatch_tool fork 子 Agent
```

因为：

```text
召回更多
+
属性提炼更多
+
内部调用链更长
+
上下文更大
```

---

# 十八、与 ItemPicker 的协作

CategoryInsight 最重要的下游消费者是：

```text
ItemPicker
```

两者字段对应：

| ItemPicker 判断 | CategoryInsight 字段 |
|---|---|
| 套装有没有缺组件 | components |
| 属性是不是主流 | attributes |
| 价格是否合理 | price_tiers |
| 是否相信知识库 | confidence |

例如：

```text
用户要旅行三件套
```

候选商品只有：

```text
洗漱包
鞋包
```

但：

```python
components = [
    "洗漱包",
    "鞋包",
    "数码线收纳"
]
```

ItemPicker 就可以判断：

```text
这套商品缺少数码线收纳
```

---

# 十九、confidence 的作用

`confidence` 表示当前品类知识的整体可信度。

例如：

```text
confidence = 0.78
```

说明知识库当前结论相对可靠。

如果：

```text
confidence < 0.5
```

主 Agent 可以考虑补：

```text
WebSearch
```

作为冷启动或知识库不足时的兜底。

---

# 二十、知识库刷新

CategoryInsight 运行时只负责：

```text
读索引
```

不会现场更新知识库。

知识库刷新属于：

```text
离线任务
```

示例策略：

| 数据 | 刷新频率 |
|---|---|
| 爆款卡片 | 每周 |
| 属性图谱 | 每月 |
| 价格区间 | 每月 |

数据来源可以来自：

```text
内部销售榜
平台公开榜单
商品属性聚合
历史成交价
```

---

# 二十一、整条链路

完整流程可以理解为：

```text
用户 Query
   ↓
Planner
   ↓
识别 category
   ↓
CategoryInsight
   ↓
Query Tower
   ↓
OpenSearch
   ├── KNN
   └── BM25
   ↓
Hybrid Top-K
   ↓
按 card_type 分组
   ↓
提炼
   ├── components
   ├── bestsellers
   ├── attributes
   └── price_tiers
   ↓
摘要
   ↓
CategoryInsightOutput
   ↓
ItemSearch / ItemPicker
```

如果使用 fork：

```text
主 Agent
   ↓
dispatch_tool
   ↓
子 Agent 完成整个 RAG 链路
   ↓
只返回 CategoryInsightOutput
```

---

# 二十二、本章最重要的几个结论

### 1. CategoryInsight 是“认知工具”，不是“搜索工具”

```text
ItemSearch：找商品
CategoryInsight：理解品类
```

二者职责不同。

### 2. 商品 RAG 不应该直接堆原始文章

更合理的是先构建：

```text
爆款卡片
属性卡片
价格卡片
```

让在线检索拿到的是高密度知识。

### 3. RAG 是三步，而不是一步

```text
召回
→ 提炼
→ 摘要
```

Vector Search 只是第一步。

### 4. Hybrid Retrieval 比纯向量检索更稳

```text
KNN
+
BM25
```

分别解决：

```text
语义匹配
+
关键词精确匹配
```

### 5. 工具最终返回结论，不返回原文

```text
raw evidence
→ 工具内部消化
→ structured insight
```

这也是 Agent 工程中控制上下文的重要方式。

### 6. CategoryInsight 是典型的 fork 候选

尤其是：

```text
depth = deep
```

因为它同时满足：

```text
调用链较长
+
中间上下文需要隔离
```

### 7. CategoryInsight 最终服务于 ItemPicker

ItemPicker 判断商品是否“值得选”，不能只看商品自身字段，还需要：

```text
品类常识
+
主流属性
+
价格区间
+
组件完整性
```

而这些信息正是 CategoryInsight 提供的。

---

# 一句话总结

> **CategoryInsight 本质上是 Globex 的“品类知识 RAG 子 Agent”：它通过 OpenSearch Hybrid Retrieval 召回商品知识卡片，再经过提炼和摘要，把原始知识压缩成 components、bestsellers、attributes、price_tiers 和 confidence，供 ItemSearch 与 ItemPicker 做更懂行的搜索和决策。**
