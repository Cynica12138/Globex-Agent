# -*- coding: utf-8 -*-
"""category_insight_tool

品类洞察工具（RAG）：回答"这个品类当前热卖什么、看哪些属性、价格区间、有什么坑"
这类选购常识问题，与 product_search_tool（出具体商品清单）分工明确。

注意：本模块不能用 `from __future__ import annotations`（AgentScope schema 生成依赖运行时注解）。
"""
import json
from datetime import datetime, timezone

from agentscope.message import TextBlock, ToolResultState
from agentscope.rag import KnowledgeBase
from agentscope.tool import ToolChunk

from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


def _chunk_text(content) -> str:
    """Chunk.content 是 TextBlock / DataBlock 而非纯字符串，统一归一为可序列化文本。"""
    if isinstance(content, str):
        return content
    text = getattr(content, "text", None)
    if text is not None:
        return text
    if isinstance(content, dict):
        return content.get("text") or str(content)
    return str(content)


def build_category_insight_tool(
    knowledge_base: KnowledgeBase,
    bus: TradeEventBus,
    min_score: float = 0.35,
):
    async def category_insight_tool(question: str, top_k: int = 3) -> ToolChunk:
        """查询品类洞察知识库：热卖款型、关键属性判断口径、价格区间、避坑点、跨境通则。

        适用于"这个品类怎么挑""现在流行什么""多少钱算合理""有什么坑"这类选购常识问题；
        需要具体商品清单与价格时用 product_search_tool。

        Args:
            question (`str`):
                自然语言问题，建议带上品类词，如"旅行装备怎么挑材质"、"美国免税额度多少"。
            top_k (`int`):
                返回知识片段数量，默认 3。
        """
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {"tool": "category_insight_tool", "args": {"question": question, "top_k": top_k}},
        )
        try:
            results = await knowledge_base.search(queries=[question], top_k=top_k)
        except Exception as err:  # noqa: BLE001 —— 知识库不可用时如实降级，不编造洞察
            bus.publish(session_id, "tool.result", {"tool": "category_insight_tool", "error": str(err)})
            return ToolChunk(
                content=[TextBlock(type="text", text=f"[error] 品类知识库不可用：{err}")],
                state=ToolResultState.ERROR,
            )

        insights = []
        for item in results:
            if item.score < min_score:
                continue
            metadata = item.chunk.metadata or {}
            insights.append(
                {
                    "content": _chunk_text(item.chunk.content),
                    "source": metadata.get("source", item.document_id),
                    "score": round(item.score, 4),
                    "source_type": metadata.get("source_type", "demo_reference"),
                    "source_version": metadata.get("source_version", "unknown"),
                    "source_updated_at": metadata.get("source_updated_at", "unknown"),
                    "effective_at": metadata.get("effective_at", "not_applicable"),
                    "expires_at": metadata.get("expires_at", "not_applicable"),
                    "freshness_policy": metadata.get("freshness_policy", "manual_review"),
                    "is_realtime": bool(metadata.get("is_realtime", False)),
                },
            )

        best_score = max((item["score"] for item in insights), default=0.0)
        confidence = "high" if best_score >= 0.75 else ("medium" if insights else "low")
        payload = {
            "insights": insights,
            "answerable": bool(insights),
            "confidence": confidence,
            "min_score": min_score,
            "data_mode": "demo_reference",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "freshness_warning": (
                "这些是本地 Demo 选购参考，不是实时法规源。"
                "关税、免税额度、禁限运和航空规则需通过联网工具或官方源复核。"
            ),
        }
        if not insights:
            payload["reason"] = "no_reliable_local_evidence"
            payload["suggested_action"] = (
                "请缩小或改写问题；若涉及实时法规，应调用联网搜索并优先核对官方来源。"
            )
        bus.publish(
            session_id,
            "tool.result",
            {
                "tool": "category_insight_tool",
                "hit_count": len(insights),
                "answerable": bool(insights),
                "confidence": confidence,
            },
        )
        return ToolChunk(
            content=[TextBlock(type="text", text=json.dumps(payload, ensure_ascii=False))],
            state=ToolResultState.SUCCESS,
        )

    return category_insight_tool
