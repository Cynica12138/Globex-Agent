# -*- coding: utf-8 -*-
"""Product 聚合根

Globex 把跨境商品建模为 Product（SPU）+ Sku（多个），携带品牌、产地、亮点等结构化属性。
SearchAgent 召回的"候选集"传递的就是 Product 卡片，TradeAgent 创建订单时再以 Sku 粒度结算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from app.domain.catalog.sku import Sku


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProductStatus(str, Enum):
    """商品在交易事实库中的状态。

    Demo 只保留最小的在售/下架两态，避免把平台审核、预售等复杂状态
    过早搬进项目。
    """

    ACTIVE = "ACTIVE"
    OFF_SHELF = "OFF_SHELF"


@dataclass(frozen=True)
class ProductHighlight:
    label: str
    detail: str = ""


@dataclass
class Product:
    product_id: str
    title: str
    brand: str
    category: str
    origin_country: str
    description: str
    highlights: list[ProductHighlight] = field(default_factory=list)
    ships_to: list[str] = field(default_factory=list)
    skus: list[Sku] = field(default_factory=list)
    status: ProductStatus = ProductStatus.ACTIVE
    source: str = "demo_seed"
    source_updated_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not self.product_id:
            raise ValueError("Product.product_id required")
        if not self.skus:
            raise ValueError(f"Product 至少要有一个 Sku：{self.product_id}")
        if not isinstance(self.status, ProductStatus):
            self.status = ProductStatus(self.status)

    def primary_sku(self) -> Sku:
        return self.skus[0]

    def is_active(self) -> bool:
        return self.status is ProductStatus.ACTIVE

    def find_sku(self, sku_id: str) -> Optional[Sku]:
        return next((s for s in self.skus if s.sku_id == sku_id), None)

    def searchable_text(self) -> str:
        """召回用的可检索文本：标题 + 品牌 + 品类 + 描述 + 亮点。"""
        highlight_text = " ".join(f"{h.label} {h.detail}" for h in self.highlights)
        return " ".join(
            [self.title, self.brand, self.category, self.origin_country, self.description, highlight_text],
        )
