# -*- coding: utf-8 -*-
"""category_knowledge

品类洞察 RAG 知识库：复用 AgentScope 2.0 的 KnowledgeBase + QdrantStore + OpenAIEmbeddingModel。

与商品向量索引分开两套 collection：
    globex_products     商品卡向量（模块一：二阶段召回）
    globex_category_kb  品类洞察知识（本模块：RAG 问答）

建库流程：TextParser 读 knowledge/*.md → ApproxTokenChunker 切块 → insert_document（按文件名做 document_id，幂等）。
"""
from __future__ import annotations

import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agentscope.credential import OpenAICredential
from agentscope.embedding import OpenAIEmbeddingModel
from agentscope.rag import ApproxTokenChunker, KnowledgeBase, QdrantStore, TextParser

from app.infrastructure.settings import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

_KB_DESCRIPTION = (
    "Globex 跨境电商品类洞察知识库：各品类的热卖款型、关键属性判断口径、"
    "价格区间参考、避坑点，以及跨境到手价/免税额度/合规通则。"
)


def build_source_metadata(path: Path) -> dict:
    """为本地知识文档生成可追溯元数据，不虚构法规生效日期。"""
    content = path.read_bytes()
    source_updated_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    return {
        "source": path.name,
        "source_type": "demo_reference",
        "source_version": hashlib.sha256(content).hexdigest()[:12],
        "source_updated_at": source_updated_at,
        "effective_at": "not_applicable",
        "expires_at": "not_applicable",
        "freshness_policy": "manual_review",
        "is_realtime": False,
        "category": path.stem,
    }


def build_category_knowledge_base(settings: Settings) -> KnowledgeBase:
    """构建品类知识库对象（不建库，建库见 bootstrap_category_knowledge）。"""
    credential = OpenAICredential(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )
    embedding_model = OpenAIEmbeddingModel(
        credential=credential,
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
        pass_dimensions=False,  # 兼容不接受 dimensions 入参的网关，维度仅用于建 collection
    )
    if settings.qdrant_url:
        vector_store = QdrantStore(url=settings.qdrant_url)
    else:
        local_path = settings.data_dir / "qdrant_kb"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store = QdrantStore(path=str(local_path))
    return KnowledgeBase(
        name="category_insight",
        description=_KB_DESCRIPTION,
        embedding_model=embedding_model,
        vector_store=vector_store,
        collection=settings.category_kb_collection,
    )


async def bootstrap_category_knowledge(
    knowledge_base: KnowledgeBase,
    knowledge_dir: Optional[Path] = None,
) -> int:
    """把 knowledge/*.md 增量灌入知识库，内容版本变化时重建；失败仅告警返回 0。"""
    directory = knowledge_dir or KNOWLEDGE_DIR
    try:
        await knowledge_base.ensure_collection()
        existing = {doc.document_id: doc for doc in await knowledge_base.list_documents()}
        parser, chunker = TextParser(), ApproxTokenChunker(chunk_size=512, overlap=50)
        upserted = 0
        newly_inserted = 0
        for md_file in sorted(directory.glob("*.md")):
            document_id = md_file.stem
            metadata = build_source_metadata(md_file)
            indexed = existing.get(document_id)
            if indexed and indexed.metadata.get("source_version") == metadata["source_version"]:
                continue
            if indexed:
                # 老索引没有版本字段，或文件内容已变化：整篇删除后重新切片与向量化。
                await knowledge_base.delete_document(document_id)
            else:
                newly_inserted += 1
            sections = await parser.parse(str(md_file), filename=md_file.name)
            chunks = await chunker.chunk(sections)
            await knowledge_base.insert_document(
                chunks=chunks,
                document_id=document_id,
                document_metadata=metadata,
            )
            upserted += 1
        logger.info(
            "品类知识库就绪：新增/更新 %d 篇，累计 %d 篇",
            upserted,
            len(existing) + newly_inserted,
        )
        return upserted
    except Exception as err:  # noqa: BLE001 —— 知识库不可用不阻塞启动
        logger.warning("品类知识库建库失败，category_insight 将不可用：%s", err)
        return 0
