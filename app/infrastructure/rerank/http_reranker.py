# -*- coding: utf-8 -*-
"""HttpReranker

HTTP 精排客户端（对接 Hugging Face TEI 的 /rerank 协议服务）。
RERANKER_BASE_URL 未配置时组装根不会实例化本类；调用失败抛异常，
由 CatalogSearchUseCase 降级为按向量分排序并标注 rerank_applied=false。
"""
from __future__ import annotations

import httpx

from app.domain.catalog.ports.retrieval_ports import Reranker
from app.infrastructure.settings import Settings


class HttpReranker(Reranker):
    def __init__(
        self,
        settings: Settings,
        timeout_seconds: float = 3.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = settings.reranker_base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            response = await client.post(
                f"{self._base_url}/rerank",
                # TEI 在服务启动时已绑定模型，请求体字段名是 texts，不发送 model/documents。
                json={"query": query, "texts": documents},
            )
            response.raise_for_status()
            body = response.json()
        # TEI 返回列表；同时兼容部分网关包一层 {results:[...]} 的形态。
        results = body if isinstance(body, list) else body.get("results")
        if not isinstance(results, list) or len(results) != len(documents):
            raise RuntimeError(f"rerank 响应异常：{str(body)[:200]}")
        scores = [0.0] * len(documents)
        for item in results:
            index = item.get("index")
            if not isinstance(index, int) or not 0 <= index < len(documents):
                raise RuntimeError(f"rerank 响应 index 越界：{str(item)[:100]}")
            scores[index] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores
