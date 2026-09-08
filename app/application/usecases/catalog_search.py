# -*- coding: utf-8 -*-
"""CatalogSearchUseCase

商品检索核心 UseCase：
    1. 向量召回与本地 BM25 各取一批候选
    2. RRF 融合两路排名，避免直接混加不可比的分数
    3. 先执行上下架 / 配送 / 价格硬约束
    4. 可选 Cross-Encoder Reranker 精排；不可用时保留 RRF 排序
    5. 组装商品卡 JSON；命中 ship_to 时内联到手价

降级链（recall_strategy 如实标注）：
    hybrid_rrf_rerank → hybrid_rrf → embedding_only → bm25

计价收敛设计：到手价在检索链路内联计算（TariffSchedule 规则内核），
不给 Agent 单独暴露比价/运费工具，减少不必要的工具调用轮次。

过滤可观测：被 ship_to / price_max_major 硬约束挡掉的候选以 filtered_out 摘要回传，
让模型能区分"库里没有这个商品"与"有但不满足约束"，不致于给出误导性结论。
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.ports.retrieval_ports import (
    EmbeddingClient,
    ProductVectorIndex,
    Reranker,
)
from app.domain.catalog.product import Product
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.shipping.tariff_schedule import TariffSchedule

logger = logging.getLogger(__name__)

# 一阶段召回候选数（显著大于最终 top_k，给融合与精排留空间）
_RECALL_TOP_N = 30

# RRF 常用平滑常数，降低某一路第一名对融合结果的过度支配
_RRF_K = 60

# 商品卡都很短，标准 b=0.75 会过度奖励更短但缺关键词的文档；评测语料上取 0.6。
_BM25_B = 0.6

# 被硬约束挡掉的候选回传条数上限（只回摘要，避免上下文膨胀）
_FILTERED_OUT_LIMIT = 3


@dataclass(frozen=True)
class ProductCard:
    product_id: str
    title: str
    brand: str
    category: str
    origin_country: str
    price_major: float
    currency: str
    highlights: list[str]
    skus: list[dict]
    score: float
    landed_price: Optional[dict]  # ship_to 命中时的到手价明细，未命中为 None
    data_mode: str
    source: str
    source_updated_at: str

    def to_dict(self) -> dict:
        card = {
            "product_id": self.product_id,
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "origin_country": self.origin_country,
            "price_major": self.price_major,
            "currency": self.currency,
            "highlights": self.highlights,
            "skus": self.skus,
            "score": round(self.score, 4),
            "data_mode": self.data_mode,
            "source": self.source,
            "source_updated_at": self.source_updated_at,
        }
        if self.landed_price is not None:
            card["landed_price"] = self.landed_price
        return card


def tokenize(text: str) -> list[str]:
    """轻量中英分词：英文/数字按词，中文保留连续片段并补 2-gram。"""
    terms: list[str] = []
    for chunk in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        terms.append(chunk)
        # 两字词本身就是完整 token，不再重复加入同一个 2-gram。
        if any("\u4e00" <= ch <= "\u9fff" for ch in chunk) and len(chunk) > 2:
            terms.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return terms


class CatalogSearchUseCase:
    def __init__(
        self,
        product_repo: ProductRepository,
        embedder: Optional[EmbeddingClient] = None,
        vector_index: Optional[ProductVectorIndex] = None,
        reranker: Optional[Reranker] = None,
        tariff_schedule: Optional[TariffSchedule] = None,
        recall_top_n: int = _RECALL_TOP_N,
        hybrid_enabled: bool = True,
        rrf_k: int = _RRF_K,
    ) -> None:
        self._product_repo = product_repo
        self._embedder = embedder
        self._vector_index = vector_index
        self._reranker = reranker
        self._tariff = tariff_schedule or TariffSchedule(rates=ExchangeRateTable())
        self._recall_top_n = max(1, recall_top_n)
        self._hybrid_enabled = hybrid_enabled
        self._rrf_k = max(1, rrf_k)

    async def execute(self, spec: ProductSearchSpec) -> dict:
        observed_at = datetime.now(timezone.utc).isoformat()
        vector_scored: list[tuple[float, Product]] = []
        bm25_scored: list[tuple[float, Product]] = []
        scored: list[tuple[float, Product]] = []
        recall_strategy = "bm25"
        rerank_applied = False

        if self._embedder is not None and self._vector_index is not None:
            try:
                vector_scored = await self._vector_recall(spec)
                recall_strategy = "embedding_only"
            except Exception as err:  # noqa: BLE001 —— 召回基建异常必须降级而非失败
                logger.warning("向量召回不可用，降级 BM25 召回：%s", err)
                vector_scored = []

        if self._hybrid_enabled:
            bm25_scored = await self._bm25_recall(spec)
            if vector_scored:
                scored = self._rrf_fuse(vector_scored, bm25_scored)
                recall_strategy = "hybrid_rrf"
            else:
                scored = bm25_scored
                recall_strategy = "bm25"
        else:
            scored = vector_scored
            if not scored:
                bm25_scored = await self._bm25_recall(spec)
                scored = bm25_scored
                recall_strategy = "bm25"

        # 硬约束先于精排：不把下架、不可配送、超预算商品送给外部 Reranker。
        eligible: list[tuple[float, Product]] = []
        filtered_out: list[dict] = []
        for score, product in scored:
            reason = self._reject_reason(product, spec)
            if reason is None:
                eligible.append((score, product))
            elif len(filtered_out) < _FILTERED_OUT_LIMIT:
                filtered_out.append(self._to_rejected(product, spec, reason))

        if eligible and self._reranker is not None:
            try:
                eligible = await self._rerank(spec, eligible)
                recall_strategy = (
                    "hybrid_rrf_rerank" if recall_strategy == "hybrid_rrf" else "embedding_rerank"
                )
                rerank_applied = True
            except Exception as err:  # noqa: BLE001
                logger.warning("rerank 不可用，保留一阶段排序：%s", err)

        hits = [self._to_card(score, product, spec) for score, product in eligible[: spec.top_k]]
        result = {
            "hits": [card.to_dict() for card in hits],
            "total_candidates": len(eligible),
            "recall_strategy": recall_strategy,
            "rerank_applied": rerank_applied,
            "retrieval_debug": {
                "recall_top_n": self._recall_top_n,
                "vector_candidates": len(vector_scored),
                "bm25_candidates": len(bm25_scored),
                "fused_candidates": len(scored),
                "eligible_candidates": len(eligible),
            },
            # 明确这是本地模拟交易事实，不伪装成真实电商平台实时价。
            "data_mode": "demo",
            "observed_at": observed_at,
        }
        if filtered_out:
            # 如实告知"召回到了但被硬约束挡掉"，否则模型分不清"库里没有"与"被过滤"，
            # 会把超预算商品答成"没有这个商品"
            result["filtered_out"] = filtered_out
        return result

    def _reject_reason(self, product: Product, spec: ProductSearchSpec) -> Optional[str]:
        """返回硬约束拒绝原因，None 表示通过。"""
        if not product.is_active():
            return "off_shelf"
        if spec.ship_to and spec.ship_to not in product.ships_to:
            return "ship_to_unavailable"
        if not self._within_price_cap(product, spec):
            return "over_price_cap"
        return None

    def _to_rejected(self, product: Product, spec: ProductSearchSpec, reason: str) -> dict:
        primary_in_target = self._tariff.rates.convert(product.primary_sku().price, spec.target_currency)
        return {
            "product_id": product.product_id,
            "title": product.title,
            "category": product.category,
            "price_major": round(primary_in_target.to_major_units(), 2),
            "currency": spec.target_currency,
            "reason": reason,
        }

    def _within_price_cap(self, product: Product, spec: ProductSearchSpec) -> bool:
        if spec.price_max_major is None:
            return True
        primary_in_target = self._tariff.rates.convert(product.primary_sku().price, spec.target_currency)
        return primary_in_target.to_major_units() <= spec.price_max_major

    # ---- 一阶段：向量召回 ----

    async def _vector_recall(self, spec: ProductSearchSpec) -> list[tuple[float, Product]]:
        embedding = await self._embedder.embed(spec.normalized_query)
        vector_hits = await self._vector_index.search(embedding, top_n=self._recall_top_n)
        products = await self._product_repo.find_by_ids([hit.product_id for hit in vector_hits])
        by_id = {product.product_id: product for product in products}
        return [
            (hit.score, by_id[hit.product_id])
            for hit in vector_hits
            if hit.product_id in by_id
        ]

    # ---- 二阶段：精排 ----

    async def _rerank(
        self,
        spec: ProductSearchSpec,
        scored: list[tuple[float, Product]],
    ) -> list[tuple[float, Product]]:
        if self._reranker is None:
            raise RuntimeError("Reranker 未配置")
        documents = [product.searchable_text() for _, product in scored]
        rerank_scores = await self._reranker.rerank(spec.normalized_query, documents)
        reranked = [
            (rerank_scores[i], product)
            for i, (_, product) in enumerate(scored)
        ]
        reranked.sort(key=lambda pair: pair[0], reverse=True)
        return reranked

    # ---- 一阶段：轻量 BM25 与 RRF 融合 ----

    async def _bm25_recall(self, spec: ProductSearchSpec) -> list[tuple[float, Product]]:
        products = await self._product_repo.list_all()
        if not products:
            return []
        query_terms = tokenize(spec.normalized_query)
        if spec.category:
            query_terms.extend(tokenize(spec.category))
        if not query_terms:
            return []

        documents = [tokenize(product.searchable_text()) for product in products]
        doc_freq = Counter(term for terms in documents for term in set(terms))
        avg_len = sum(len(terms) for terms in documents) / len(documents)
        query_freq = Counter(query_terms)
        candidates: list[tuple[float, Product]] = []
        for product, terms in zip(products, documents):
            frequencies = Counter(terms)
            score = 0.0
            for term, qf in query_freq.items():
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1.0 + (len(documents) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                norm = tf + 1.5 * (1.0 - _BM25_B + _BM25_B * len(terms) / max(avg_len, 1.0))
                score += idf * (tf * 2.5 / norm) * qf
            # 中文商品 query 往往是多个短属性的 AND 意图；协调因子抑制只重复命中
            # 热门品类词、却遗漏关键属性词的短文档。
            matched_terms = sum(1 for term in query_freq if frequencies.get(term, 0))
            score *= (matched_terms / len(query_freq)) ** 2
            if spec.category and spec.category in product.category:
                score += 1.0
            if score > 0:
                candidates.append((score, product))
        candidates.sort(key=lambda pair: (-pair[0], pair[1].product_id))
        return candidates[: self._recall_top_n]

    def _rrf_fuse(
        self,
        vector_scored: list[tuple[float, Product]],
        bm25_scored: list[tuple[float, Product]],
    ) -> list[tuple[float, Product]]:
        """Reciprocal Rank Fusion：只使用名次，避免向量分与 BM25 分尺度不一致。"""
        products: dict[str, Product] = {}
        scores: Counter[str] = Counter()
        for ranking in (vector_scored, bm25_scored):
            for rank, (_, product) in enumerate(ranking, start=1):
                products[product.product_id] = product
                scores[product.product_id] += 1.0 / (self._rrf_k + rank)
        fused = [(score, products[product_id]) for product_id, score in scores.items()]
        fused.sort(key=lambda pair: (-pair[0], pair[1].product_id))
        return fused[: self._recall_top_n]

    # ---- 商品卡组装（含到手价内联）----

    def _to_card(self, score: float, product: Product, spec: ProductSearchSpec) -> ProductCard:
        primary = product.primary_sku()
        landed_price: Optional[dict] = None
        if spec.ship_to:
            try:
                quote = self._tariff.quote(
                    subtotal=primary.price,
                    category=product.category,
                    ship_to=spec.ship_to,
                    quantity=1,
                    target_currency=spec.target_currency,
                )
                landed_price = quote.to_dict()
            except ValueError as err:
                # 目的国不在规则表内：如实标注，不编造数字
                landed_price = {"unavailable_reason": str(err)}
        return ProductCard(
            product_id=product.product_id,
            title=product.title,
            brand=product.brand,
            category=product.category,
            origin_country=product.origin_country,
            price_major=primary.price.to_major_units(),
            currency=primary.price.currency,
            highlights=[f"{h.label}：{h.detail}" if h.detail else h.label for h in product.highlights],
            skus=[
                {
                    "sku_id": sku.sku_id,
                    "spec": sku.spec,
                    "price_major": sku.price.to_major_units(),
                    "currency": sku.price.currency,
                    "stock": sku.stock,
                    "version": sku.version,
                    "price_updated_at": sku.price_updated_at,
                    "stock_updated_at": sku.stock_updated_at,
                }
                for sku in product.skus
            ],
            score=score,
            landed_price=landed_price,
            data_mode="demo",
            source=product.source,
            source_updated_at=product.source_updated_at,
        )
