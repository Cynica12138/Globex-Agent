# -*- coding: utf-8 -*-
"""关系库持久化实现（SQLAlchemy 2.0 async）

当前只验证与交付 sqlite+aiosqlite（零外部依赖，开箱即用）。
要换 MySQL / PostgreSQL：装上对应异步驱动（aiomysql / asyncpg）并把 DATABASE_URL
改成该驱动即可，仓储代码不需要改；但本仓未验证过那些驱动的特有行为。

实现五个领域端口：ProductRepository / SessionStore / ConversationStore /
OrderRepository / PreferenceStore。
domain 与 application 不感知本模块的存在，替换存储只改组装根。

并发安全要点：
    - 订单保存用 merge 覆盖写（订单号唯一，状态机由 domain 保证合法迁移）
    - 偏好去重靠唯一约束，重复插入吞掉 IntegrityError（比先查后插更可靠）
    - turn_index 按会话取当前最大值 +1，同会话并发写有极小概率撞号，
      撞号只影响展示顺序不影响数据完整性，故不加分布式锁

SQLite 的边界（重要）：单写者模型。模块三的 worker 是独立进程，与 API 进程并发写
同一个 db 文件时可能碰到 "database is locked"；WAL 模式能缓解，高并发仍应换服务型数据库。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, event, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.domain.buyer.preference import BuyerPreference, PreferenceStore
from app.domain.catalog.money import Money
from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.product import Product, ProductHighlight, ProductStatus
from app.domain.catalog.sku import Sku
from app.domain.order.address import Address
from app.domain.order.order import Order, OrderStatus
from app.domain.order.order_line import OrderLine
from app.domain.order.ports.order_repository import OrderRepository
from app.domain.session.ports.conversation_store import (
    ConversationEventRecord,
    ConversationStore,
    ConversationTurn,
)
from app.domain.session.ports.session_store import SessionStore
from app.infrastructure.persistence.sql.tables import (
    AgentSessionStateRow,
    Base,
    BuyerPreferenceRow,
    ConversationEventRow,
    ConversationMessageRow,
    ConversationSessionRow,
    OrderLineRow,
    OrderRow,
    ProductRow,
    SkuRow,
)

logger = logging.getLogger(__name__)


def create_engine(database_url: str) -> AsyncEngine:
    """创建异步引擎。连接池参数必须按驱动分开给。

    SQLite：不能传 pool_size / max_overflow（对其默认池无意义），pool_recycle 也无处可用
    （本地文件连接不会被服务端回收）。开 WAL 让读写不互斥，缓解 worker 与 API
    双进程并发写时的 "database is locked"。
    服务型数据库：必需 pool_pre_ping，否则空闲连接被服务端回收后首次查询必报断连。
    """
    if database_url.startswith("sqlite"):
        engine = create_async_engine(database_url, echo=False)

        @event.listens_for(engine.sync_engine, "connect")
        def _enable_wal(dbapi_conn, _record):  # noqa: ANN001
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")  # 锁竞争时等待而不是立即报错
            cursor.close()

        return engine
    return create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )


async def bootstrap_schema(engine: AsyncEngine) -> None:
    """幂等建表。生产环境应改用 Alembic 迁移。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据库表结构已就绪（%s）", engine.url.get_backend_name())


class SqlProductRepository(ProductRepository):
    """商品/SKU 事实库。

    Qdrant 只返回 product_id；价格、库存和状态均在这里读取。
    reserve_stock 用带库存下限条件的 UPDATE，两个并发下单不会同时卖掉
    最后一件库存。
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def find_by_id(self, product_id: str) -> Optional[Product]:
        async with self._session_factory() as db:
            return await _load_product(db, product_id)

    async def find_by_ids(self, product_ids: list[str]) -> list[Product]:
        if not product_ids:
            return []
        async with self._session_factory() as db:
            rows = (
                await db.scalars(select(ProductRow).where(ProductRow.product_id.in_(product_ids)))
            ).all()
            found = {
                row.product_id: await _product_row_to_domain(db, row)
                for row in rows
            }
        # 保持向量召回给出的顺序，否则 score 会与商品错位
        return [found[product_id] for product_id in product_ids if product_id in found]

    async def list_all(self) -> list[Product]:
        async with self._session_factory() as db:
            rows = (
                await db.scalars(
                    select(ProductRow)
                    .where(ProductRow.status == ProductStatus.ACTIVE.value)
                    .order_by(ProductRow.product_id),
                )
            ).all()
            return [await _product_row_to_domain(db, row) for row in rows]

    async def reserve_stock(
        self,
        product_id: str,
        sku_id: str,
        quantity: int,
        expected_version: Optional[int] = None,
    ) -> tuple[Product, Sku]:
        if quantity <= 0:
            raise ValueError("quantity 必须为正整数")
        now = datetime.now(timezone.utc)
        async with self._session_factory() as db:
            product_row = await db.get(ProductRow, product_id)
            if product_row is None:
                raise ValueError(f"商品不存在：{product_id}")
            if product_row.status != ProductStatus.ACTIVE.value:
                raise ValueError(f"商品已下架：{product_id}")
            sku_row = await db.get(SkuRow, sku_id)
            if sku_row is None or sku_row.product_id != product_id:
                raise ValueError(f"Sku 不存在：{product_id}/{sku_id}")
            if expected_version is not None and sku_row.version != expected_version:
                raise ValueError(
                    f"SKU 信息已变化：{sku_id}，快照 v{expected_version}，当前 v{sku_row.version}；"
                    "请重新搜索并让买家再次确认",
                )

            result = await db.execute(
                update(SkuRow)
                .where(
                    SkuRow.sku_id == sku_id,
                    SkuRow.product_id == product_id,
                    SkuRow.stock >= quantity,
                    SkuRow.product_id.in_(
                        select(ProductRow.product_id).where(
                            ProductRow.status == ProductStatus.ACTIVE.value,
                        ),
                    ),
                    *(
                        (SkuRow.version == expected_version,)
                        if expected_version is not None
                        else ()
                    ),
                )
                .values(
                    stock=SkuRow.stock - quantity,
                    version=SkuRow.version + 1,
                    stock_updated_at=now,
                ),
            )
            if result.rowcount != 1:
                await db.refresh(product_row)
                await db.refresh(sku_row)
                if product_row.status != ProductStatus.ACTIVE.value:
                    raise ValueError(f"商品已下架：{product_id}")
                if expected_version is not None and sku_row.version != expected_version:
                    raise ValueError(
                        f"SKU 信息已变化：{sku_id}，快照 v{expected_version}，"
                        f"当前 v{sku_row.version}；请重新搜索并让买家再次确认",
                    )
                raise ValueError(
                    f"Sku 库存不足：{sku_id}，剩余 {sku_row.stock}，需要 {quantity}",
                )
            product_row.source_updated_at = now
            await db.flush()
            await db.refresh(sku_row)
            product = await _product_row_to_domain(db, product_row)
            sku = product.find_sku(sku_id)
            assert sku is not None
            await db.commit()
            return product, sku

    async def restore_stock(self, product_id: str, sku_id: str, quantity: int) -> None:
        if quantity < 0:
            raise ValueError("restore_stock.quantity 必须非负")
        now = datetime.now(timezone.utc)
        async with self._session_factory() as db:
            result = await db.execute(
                update(SkuRow)
                .where(SkuRow.sku_id == sku_id, SkuRow.product_id == product_id)
                .values(
                    stock=SkuRow.stock + quantity,
                    version=SkuRow.version + 1,
                    stock_updated_at=now,
                ),
            )
            if result.rowcount != 1:
                raise ValueError(f"Sku 不存在：{product_id}/{sku_id}")
            await db.execute(
                update(ProductRow)
                .where(ProductRow.product_id == product_id)
                .values(source_updated_at=now),
            )
            await db.commit()

    async def update_sku_price(self, product_id: str, sku_id: str, price: Money) -> Product:
        now = datetime.now(timezone.utc)
        async with self._session_factory() as db:
            result = await db.execute(
                update(SkuRow)
                .where(SkuRow.sku_id == sku_id, SkuRow.product_id == product_id)
                .values(
                    price_minor=price.amount_in_minor_units,
                    currency=price.currency,
                    version=SkuRow.version + 1,
                    price_updated_at=now,
                ),
            )
            if result.rowcount != 1:
                raise ValueError(f"Sku 不存在：{product_id}/{sku_id}")
            await db.execute(
                update(ProductRow)
                .where(ProductRow.product_id == product_id)
                .values(source="demo_scenario", source_updated_at=now),
            )
            await db.flush()
            product = await _load_product(db, product_id)
            assert product is not None
            await db.commit()
            return product

    async def update_sku_stock(self, product_id: str, sku_id: str, stock: int) -> Product:
        if stock < 0:
            raise ValueError("stock 必须非负")
        now = datetime.now(timezone.utc)
        async with self._session_factory() as db:
            result = await db.execute(
                update(SkuRow)
                .where(SkuRow.sku_id == sku_id, SkuRow.product_id == product_id)
                .values(stock=stock, version=SkuRow.version + 1, stock_updated_at=now)
            )
            if result.rowcount != 1:
                raise ValueError(f"Sku 不存在：{product_id}/{sku_id}")
            await db.execute(
                update(ProductRow)
                .where(ProductRow.product_id == product_id)
                .values(source="demo_scenario", source_updated_at=now),
            )
            await db.flush()
            product = await _load_product(db, product_id)
            assert product is not None
            await db.commit()
            return product

    async def update_product_status(self, product_id: str, status: ProductStatus) -> Product:
        now = datetime.now(timezone.utc)
        async with self._session_factory() as db:
            result = await db.execute(
                update(ProductRow)
                .where(ProductRow.product_id == product_id)
                .values(status=status.value, source="demo_scenario", source_updated_at=now),
            )
            if result.rowcount != 1:
                raise ValueError(f"商品不存在：{product_id}")
            await db.flush()
            product = await _load_product(db, product_id)
            assert product is not None
            await db.commit()
            return product

    async def seed_if_empty(self, products: list[Product]) -> bool:
        async with self._session_factory() as db:
            count = await db.scalar(select(func.count()).select_from(ProductRow))
            if count:
                return False
            _add_products(db, products)
            try:
                await db.commit()
            except IntegrityError:
                # API 与 worker 可能同时首次启动；另一进程已导入时按幂等成功处理。
                await db.rollback()
                return False
            return True

    async def replace_all(self, products: list[Product]) -> None:
        async with self._session_factory() as db:
            await db.execute(delete(SkuRow))
            await db.execute(delete(ProductRow))
            _add_products(db, products)
            await db.commit()


async def _load_product(db, product_id: str) -> Optional[Product]:  # noqa: ANN001
    row = await db.get(ProductRow, product_id)
    if row is None:
        return None
    return await _product_row_to_domain(db, row)


async def _product_row_to_domain(db, row: ProductRow) -> Product:  # noqa: ANN001
    sku_rows = (
        await db.scalars(
            select(SkuRow).where(SkuRow.product_id == row.product_id).order_by(SkuRow.sku_id),
        )
    ).all()
    return Product(
        product_id=row.product_id,
        title=row.title,
        brand=row.brand,
        category=row.category,
        origin_country=row.origin_country,
        description=row.description,
        highlights=[ProductHighlight(**item) for item in (row.highlights_json or [])],
        ships_to=list(row.ships_to_json or []),
        skus=[
            Sku(
                sku_id=sku.sku_id,
                spec=sku.spec,
                price=Money(
                    amount_in_minor_units=sku.price_minor,
                    currency=sku.currency,
                ),
                stock=sku.stock,
                version=sku.version,
                price_updated_at=sku.price_updated_at.isoformat(),
                stock_updated_at=sku.stock_updated_at.isoformat(),
            )
            for sku in sku_rows
        ],
        status=ProductStatus(row.status),
        source=row.source,
        source_updated_at=row.source_updated_at.isoformat(),
    )


def _add_products(db, products: list[Product]) -> None:  # noqa: ANN001
    for product in products:
        db.add(
            ProductRow(
                product_id=product.product_id,
                title=product.title,
                brand=product.brand,
                category=product.category,
                origin_country=product.origin_country,
                description=product.description,
                highlights_json=[
                    {"label": highlight.label, "detail": highlight.detail}
                    for highlight in product.highlights
                ],
                ships_to_json=list(product.ships_to),
                status=product.status.value,
                source=product.source,
                source_updated_at=_parse_iso(product.source_updated_at),
            ),
        )
        db.add_all(
            [
                SkuRow(
                    sku_id=sku.sku_id,
                    product_id=product.product_id,
                    spec=sku.spec,
                    price_minor=sku.price.amount_in_minor_units,
                    currency=sku.price.currency,
                    stock=sku.stock,
                    version=sku.version,
                    price_updated_at=_parse_iso(sku.price_updated_at),
                    stock_updated_at=_parse_iso(sku.stock_updated_at),
                )
                for sku in product.skus
            ],
        )


def _parse_iso(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


class SqlSessionStore(SessionStore):
    def __init__(self, engine: AsyncEngine) -> None:
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def save(self, session_id: str, state_json: str) -> None:
        async with self._session_factory() as db:
            await db.merge(AgentSessionStateRow(session_id=session_id, state_json=state_json))
            await db.commit()

    async def load(self, session_id: str) -> Optional[str]:
        async with self._session_factory() as db:
            row = await db.get(AgentSessionStateRow, session_id)
            return row.state_json if row else None


class SqlConversationStore(ConversationStore):
    def __init__(self, engine: AsyncEngine) -> None:
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def touch_session(self, session_id: str, buyer_id: str, locale: str, currency: str) -> None:
        async with self._session_factory() as db:
            existing = await db.get(ConversationSessionRow, session_id)
            if existing is None:
                db.add(
                    ConversationSessionRow(
                        session_id=session_id, buyer_id=buyer_id, locale=locale, currency=currency,
                    ),
                )
            else:
                existing.last_active_at = datetime.now(timezone.utc)
            await db.commit()

    async def append_turn(self, turn: ConversationTurn) -> None:
        async with self._session_factory() as db:
            max_index = await db.scalar(
                select(func.max(ConversationMessageRow.turn_index)).where(
                    ConversationMessageRow.session_id == turn.session_id,
                ),
            )
            db.add(
                ConversationMessageRow(
                    session_id=turn.session_id,
                    turn_index=(max_index or 0) + 1,
                    buyer_id=turn.buyer_id,
                    role=turn.role,
                    content=turn.content,
                    model=turn.model,
                    latency_ms=turn.latency_ms,
                ),
            )
            await db.commit()

    async def append_events(self, events: list[ConversationEventRecord]) -> None:
        if not events:
            return
        async with self._session_factory() as db:
            db.add_all(
                [
                    ConversationEventRow(
                        session_id=event.session_id,
                        type=event.type,
                        payload=event.payload,
                        occurred_at=event.occurred_at,
                    )
                    for event in events
                ],
            )
            await db.commit()

    async def list_turns(self, session_id: str, limit: int = 50) -> list[ConversationTurn]:
        async with self._session_factory() as db:
            rows = (
                await db.scalars(
                    select(ConversationMessageRow)
                    .where(ConversationMessageRow.session_id == session_id)
                    .order_by(ConversationMessageRow.turn_index)
                    .limit(limit),
                )
            ).all()
        return [
            ConversationTurn(
                session_id=row.session_id,
                buyer_id=row.buyer_id,
                role=row.role,
                content=row.content,
                model=row.model,
                latency_ms=row.latency_ms,
                created_at=row.created_at.isoformat() if row.created_at else "",
            )
            for row in rows
        ]

    async def find_session(self, session_id: str) -> Optional[dict]:
        async with self._session_factory() as db:
            row = await db.get(ConversationSessionRow, session_id)
            if row is None:
                return None
            return {
                "session_id": row.session_id,
                "buyer_id": row.buyer_id,
                "locale": row.locale,
                "currency": row.currency,
                "last_active_at": row.last_active_at.isoformat() if row.last_active_at else "",
            }


class SqlOrderRepository(OrderRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def save(self, order: Order) -> None:
        total = order.total_amount()
        async with self._session_factory() as db:
            await db.merge(
                OrderRow(
                    order_id=order.order_id,
                    buyer_id=order.buyer_id,
                    status=order.status.value,
                    currency=total.currency,
                    total_amount_minor=total.amount_in_minor_units,
                    shipping_address_json=_address_to_dict(order.shipping_address),
                    created_at=order.created_at,
                    confirmed_at=order.confirmed_at,
                    cancelled_at=order.cancelled_at,
                    cancel_reason=order.cancel_reason,
                ),
            )
            # 订单行整体重写：行数固定且量小，比逐行 diff 更简单可靠
            await db.execute(delete(OrderLineRow).where(OrderLineRow.order_id == order.order_id))
            db.add_all(
                [
                    OrderLineRow(
                        order_id=order.order_id,
                        product_id=line.product_id,
                        sku_id=line.sku_id,
                        title=line.title,
                        unit_price_minor=line.unit_price.amount_in_minor_units,
                        currency=line.unit_price.currency,
                        quantity=line.quantity,
                    )
                    for line in order.lines
                ],
            )
            await db.commit()

    async def find_by_id(self, order_id: str) -> Optional[Order]:
        async with self._session_factory() as db:
            row = await db.get(OrderRow, order_id)
            if row is None:
                return None
            line_rows = (
                await db.scalars(select(OrderLineRow).where(OrderLineRow.order_id == order_id))
            ).all()
        return _row_to_order(row, line_rows)

    async def next_order_id(self) -> str:
        """按已有订单数递增。生产应改用独立序列或雪花 ID，避免并发撞号。"""
        async with self._session_factory() as db:
            count = await db.scalar(select(func.count()).select_from(OrderRow))
        return f"GBX-{(count or 0) + 1:06d}"


class SqlPreferenceStore(PreferenceStore):
    def __init__(self, engine: AsyncEngine) -> None:
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def append(self, preference: BuyerPreference) -> None:
        async with self._session_factory() as db:
            db.add(
                BuyerPreferenceRow(
                    buyer_id=preference.buyer_id,
                    kind=preference.kind,
                    statement=preference.statement,
                    created_at=preference.created_at,
                ),
            )
            try:
                await db.commit()
            except IntegrityError:
                # 唯一约束命中 = 该偏好已存在，幂等语义下静默跳过
                await db.rollback()

    async def list_by_buyer(self, buyer_id: str) -> list[BuyerPreference]:
        async with self._session_factory() as db:
            rows = (
                await db.scalars(
                    select(BuyerPreferenceRow)
                    .where(BuyerPreferenceRow.buyer_id == buyer_id)
                    .order_by(BuyerPreferenceRow.id),
                )
            ).all()
        return [
            BuyerPreference(
                buyer_id=row.buyer_id,
                kind=row.kind,
                statement=row.statement,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def delete(self, buyer_id: str, statement: str) -> bool:
        """精确匹配 statement 删除；返回是否真的删到了行。"""
        async with self._session_factory() as db:
            result = await db.execute(
                delete(BuyerPreferenceRow).where(
                    BuyerPreferenceRow.buyer_id == buyer_id,
                    BuyerPreferenceRow.statement == statement,
                ),
            )
            await db.commit()
        return bool(result.rowcount)


# ---- 领域对象 <-> 行记录转换 ----


def _address_to_dict(address: Address) -> dict:
    return {
        "recipient_name": address.recipient_name,
        "country": address.country,
        "state": address.state,
        "city": address.city,
        "address_line": address.address_line,
        "postal_code": address.postal_code,
        "phone": address.phone,
    }


def _row_to_order(row: OrderRow, line_rows: list[OrderLineRow]) -> Order:
    order = Order(
        order_id=row.order_id,
        buyer_id=row.buyer_id,
        shipping_address=Address(**row.shipping_address_json),
        lines=[
            OrderLine(
                product_id=line.product_id,
                sku_id=line.sku_id,
                title=line.title,
                unit_price=Money(amount_in_minor_units=line.unit_price_minor, currency=line.currency),
                quantity=line.quantity,
            )
            for line in line_rows
        ],
        status=OrderStatus(row.status),
        created_at=row.created_at,
        confirmed_at=row.confirmed_at,
        cancelled_at=row.cancelled_at,
        cancel_reason=row.cancel_reason,
    )
    return order
