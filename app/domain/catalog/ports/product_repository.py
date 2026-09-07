# -*- coding: utf-8 -*-
"""ProductRepository 端口

Domain 不关心实现，Infrastructure 提供内存与 SQL 两种具体仓储。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.catalog.money import Money
from app.domain.catalog.product import Product
from app.domain.catalog.product import ProductStatus
from app.domain.catalog.sku import Sku


class ProductRepository(ABC):
    @abstractmethod
    async def find_by_id(self, product_id: str) -> Optional[Product]:
        ...

    @abstractmethod
    async def find_by_ids(self, product_ids: list[str]) -> list[Product]:
        ...

    @abstractmethod
    async def list_all(self) -> list[Product]:
        ...

    @abstractmethod
    async def reserve_stock(
        self,
        product_id: str,
        sku_id: str,
        quantity: int,
        expected_version: Optional[int] = None,
    ) -> tuple[Product, Sku]:
        """下单时重新读取最新商品事实并原子预占库存。"""
        ...

    @abstractmethod
    async def restore_stock(self, product_id: str, sku_id: str, quantity: int) -> None:
        ...

    @abstractmethod
    async def update_sku_price(self, product_id: str, sku_id: str, price: Money) -> Product:
        ...

    @abstractmethod
    async def update_sku_stock(self, product_id: str, sku_id: str, stock: int) -> Product:
        ...

    @abstractmethod
    async def update_product_status(self, product_id: str, status: ProductStatus) -> Product:
        ...

    @abstractmethod
    async def seed_if_empty(self, products: list[Product]) -> bool:
        """空库才导入固定种子；导入返回 True，已有数据返回 False。"""
        ...

    @abstractmethod
    async def replace_all(self, products: list[Product]) -> None:
        """仅供 Demo 场景重置交易事实库。"""
        ...
