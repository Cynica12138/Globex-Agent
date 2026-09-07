# -*- coding: utf-8 -*-
"""InMemoryProductRepository / InMemoryOrderRepository

开发态内存仓储实现。ProductRepository 由种子数据初始化；
OrderRepository 提供自增单号（GBX-XXXX 前缀，便于日志排查）。
"""
from __future__ import annotations

import asyncio
import itertools
from datetime import datetime, timezone
from typing import Optional

from app.domain.catalog.money import Money
from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.product import Product, ProductStatus
from app.domain.catalog.sku import Sku
from app.domain.order.order import Order
from app.domain.order.ports.order_repository import OrderRepository
from app.infrastructure.persistence.seed_products import build_seed_products


class InMemoryProductRepository(ProductRepository):
    def __init__(self, products: Optional[list[Product]] = None) -> None:
        seed = products if products is not None else build_seed_products()
        self._products: dict[str, Product] = {p.product_id: p for p in seed}
        self._lock = asyncio.Lock()

    async def find_by_id(self, product_id: str) -> Optional[Product]:
        return self._products.get(product_id)

    async def find_by_ids(self, product_ids: list[str]) -> list[Product]:
        return [self._products[pid] for pid in product_ids if pid in self._products]

    async def list_all(self) -> list[Product]:
        return [product for product in self._products.values() if product.is_active()]

    async def reserve_stock(
        self,
        product_id: str,
        sku_id: str,
        quantity: int,
        expected_version: Optional[int] = None,
    ) -> tuple[Product, Sku]:
        if quantity <= 0:
            raise ValueError("quantity 必须为正整数")
        async with self._lock:
            product = self._require_product(product_id)
            if not product.is_active():
                raise ValueError(f"商品已下架：{product_id}")
            sku = self._require_sku(product, sku_id)
            if expected_version is not None and sku.version != expected_version:
                raise ValueError(
                    f"SKU 信息已变化：{sku_id}，快照 v{expected_version}，当前 v{sku.version}；"
                    "请重新搜索并让买家再次确认",
                )
            sku.deduct_stock(quantity)
            product.source_updated_at = _utc_now_iso()
            return product, sku

    async def restore_stock(self, product_id: str, sku_id: str, quantity: int) -> None:
        async with self._lock:
            product = self._require_product(product_id)
            self._require_sku(product, sku_id).restore_stock(quantity)
            product.source_updated_at = _utc_now_iso()

    async def update_sku_price(self, product_id: str, sku_id: str, price: Money) -> Product:
        async with self._lock:
            product = self._require_product(product_id)
            self._require_sku(product, sku_id).change_price(price)
            product.source = "demo_scenario"
            product.source_updated_at = _utc_now_iso()
            return product

    async def update_sku_stock(self, product_id: str, sku_id: str, stock: int) -> Product:
        async with self._lock:
            product = self._require_product(product_id)
            self._require_sku(product, sku_id).set_stock(stock)
            product.source = "demo_scenario"
            product.source_updated_at = _utc_now_iso()
            return product

    async def update_product_status(self, product_id: str, status: ProductStatus) -> Product:
        async with self._lock:
            product = self._require_product(product_id)
            product.status = status
            product.source = "demo_scenario"
            product.source_updated_at = _utc_now_iso()
            return product

    async def seed_if_empty(self, products: list[Product]) -> bool:
        async with self._lock:
            if self._products:
                return False
            self._products = {product.product_id: product for product in products}
            return True

    async def replace_all(self, products: list[Product]) -> None:
        async with self._lock:
            self._products = {product.product_id: product for product in products}

    def _require_product(self, product_id: str) -> Product:
        product = self._products.get(product_id)
        if product is None:
            raise ValueError(f"商品不存在：{product_id}")
        return product

    @staticmethod
    def _require_sku(product: Product, sku_id: str) -> Sku:
        sku = product.find_sku(sku_id)
        if sku is None:
            raise ValueError(f"Sku 不存在：{product.product_id}/{sku_id}")
        return sku


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryOrderRepository(OrderRepository):
    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._counter = itertools.count(1)

    async def save(self, order: Order) -> None:
        self._orders[order.order_id] = order

    async def find_by_id(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    async def next_order_id(self) -> str:
        return f"GBX-{next(self._counter):06d}"
