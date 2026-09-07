# -*- coding: utf-8 -*-
"""Sku 实体

Product（SPU）下的最小可售卖单元，携带价格与库存。
TradeAgent 创建订单时以 Sku 粒度结算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.domain.catalog.money import Money


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Sku:
    sku_id: str
    spec: str  # 规格描述，如 "黑色 / 20寸"
    price: Money
    stock: int
    version: int = 1
    price_updated_at: str = field(default_factory=_utc_now_iso)
    stock_updated_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not self.sku_id:
            raise ValueError("Sku.sku_id required")
        if self.stock < 0:
            raise ValueError(f"Sku.stock 必须非负：{self.sku_id}")

    def has_stock(self, quantity: int) -> bool:
        return self.stock >= quantity

    def deduct_stock(self, quantity: int) -> None:
        if not self.has_stock(quantity):
            raise ValueError(f"Sku 库存不足：{self.sku_id}，剩余 {self.stock}，需要 {quantity}")
        self.stock -= quantity
        self.version += 1
        self.stock_updated_at = _utc_now_iso()

    def restore_stock(self, quantity: int) -> None:
        if quantity < 0:
            raise ValueError("restore_stock.quantity 必须非负")
        self.stock += quantity
        self.version += 1
        self.stock_updated_at = _utc_now_iso()

    def change_price(self, price: Money) -> None:
        self.price = price
        self.version += 1
        self.price_updated_at = _utc_now_iso()

    def set_stock(self, stock: int) -> None:
        if stock < 0:
            raise ValueError(f"Sku.stock 必须非负：{self.sku_id}")
        self.stock = stock
        self.version += 1
        self.stock_updated_at = _utc_now_iso()
