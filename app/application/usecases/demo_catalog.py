# -*- coding: utf-8 -*-
"""Demo 商品事实变化。

用固定场景模拟平台同步，便于在不接真实电商 API 的情况下演示：
搜索卡是快照，价格/库存/上下架会变，下单时必须重新校验。
这些场景只改变 Demo 事实库，不写入固定的召回评测集。
"""
from __future__ import annotations

from collections.abc import Callable

from app.domain.catalog.money import Money
from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.product import Product, ProductStatus


SUPPORTED_SCENARIOS = ("price-rise", "out-of-stock", "off-shelf", "reset")
_TARGET_PRODUCT_ID = "P1001"
_TARGET_SKU_ID = "P1001-S1"


class DemoCatalogUseCase:
    def __init__(
        self,
        product_repo: ProductRepository,
        seed_factory: Callable[[], list[Product]],
    ) -> None:
        self._product_repo = product_repo
        self._seed_factory = seed_factory

    async def execute(self, scenario: str) -> dict:
        if scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(
                f"不支持的 Demo 场景：{scenario}；可选 {', '.join(SUPPORTED_SCENARIOS)}",
            )

        if scenario == "reset":
            await self._product_repo.replace_all(self._seed_factory())
            product = await self._require_target()
            return _result(scenario, product, "商品事实库已恢复为固定种子数据")

        product = await self._require_target()
        sku = product.find_sku(_TARGET_SKU_ID)
        assert sku is not None

        if scenario == "price-rise":
            new_price = Money(
                amount_in_minor_units=sku.price.amount_in_minor_units + 1000,
                currency=sku.price.currency,
            )
            product = await self._product_repo.update_sku_price(
                _TARGET_PRODUCT_ID, _TARGET_SKU_ID, new_price,
            )
            message = "模拟上游平台价格上涨 10 CNY"
        elif scenario == "out-of-stock":
            product = await self._product_repo.update_sku_stock(
                _TARGET_PRODUCT_ID, _TARGET_SKU_ID, 0,
            )
            message = "模拟上游平台的该 SKU 售罄"
        else:
            product = await self._product_repo.update_product_status(
                _TARGET_PRODUCT_ID, ProductStatus.OFF_SHELF,
            )
            message = "模拟上游平台商品下架"
        return _result(scenario, product, message)

    async def get_target(self) -> dict:
        return _snapshot(await self._require_target())

    async def _require_target(self) -> Product:
        product = await self._product_repo.find_by_id(_TARGET_PRODUCT_ID)
        if product is None:
            raise ValueError(f"Demo 目标商品不存在：{_TARGET_PRODUCT_ID}")
        return product


def _result(scenario: str, product: Product, message: str) -> dict:
    return {
        "scenario": scenario,
        "message": message,
        "data_mode": "demo",
        "product": _snapshot(product),
        "next_step": "重新搜索可看到新快照；即使使用旧卡片下单，成交前也会再次校验。",
    }


def _snapshot(product: Product) -> dict:
    return {
        "product_id": product.product_id,
        "title": product.title,
        "status": product.status.value,
        "source": product.source,
        "source_updated_at": product.source_updated_at,
        "skus": [
            {
                "sku_id": sku.sku_id,
                "price_major": sku.price.to_major_units(),
                "currency": sku.price.currency,
                "stock": sku.stock,
                "version": sku.version,
                "price_updated_at": sku.price_updated_at,
                "stock_updated_at": sku.stock_updated_at,
            }
            for sku in product.skus
        ],
    }
