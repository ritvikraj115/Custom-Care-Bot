from typing import List, Any, Optional, Iterable
import logging
import re

from langchain_core.documents import Document
try:
    from langchain_core.retrievers import BaseRetriever
except Exception:
    from langchain.schema import BaseRetriever

from app.pipeline.hierarchical_index import query_hierarchical
from app.pipeline.elasticsearch_hybrid import search_bm25_chunks
from app.pipeline.vector_store import search_website_chunks


logger = logging.getLogger(__name__)


def _preview_text(value: Any, limit: int = 180) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text)
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


class HierarchicalRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        embedder: Any,
        client_id: str,
        bot_id: str,
        top_clusters: int,
        top_chunks: int,
        exclude_chunk_refs: Optional[Iterable[str]] = None,
        precomputed_embedding: Any = None,
        source_filter: Optional[Iterable[str] | str] = None,
        enable_doc_bm25: bool = True,
        enable_social_bm25: bool = True,
    ):
        super().__init__()
        self._embedder = embedder
        self._client_id = client_id
        self._bot_id = bot_id
        self._top_clusters = top_clusters
        self._top_chunks = top_chunks
        self._exclude_chunk_refs = set(exclude_chunk_refs or [])
        self._precomputed_embedding = precomputed_embedding
        self._source_filter = source_filter
        self._enable_doc_bm25 = bool(enable_doc_bm25)
        self._enable_social_bm25 = bool(enable_social_bm25)

    def _query_embedding(self, query: str):
        if self._precomputed_embedding is not None:
            return self._precomputed_embedding
        return self._embedder.encode(query, normalize_embeddings=True)

    def _website_semantic_docs(self, query: str, top_k: int) -> List[Document]:
        try:
            emb = self._query_embedding(query)
            rows = search_website_chunks(
                emb,
                self._bot_id,
                top_k=max(1, int(top_k)),
            )
        except Exception:
            rows = []

        docs: List[Document] = []
        for idx, c in enumerate(rows):
            chunk_ref = c.get("chunk_ref") or f"website_sem_{idx}"
            if chunk_ref in self._exclude_chunk_refs:
                continue
            docs.append(
                Document(
                    page_content=c.get("text", ""),
                    metadata={
                        "chunk_ref": chunk_ref,
                        "cluster": "website",
                        "chunk_index": idx,
                        "score": c.get("score", 0.0),
                        "topic": c.get("topic") or "website",
                        "pdf": c.get("pdf"),
                        "source_type": c.get("source_type"),
                        "source_url": c.get("source_url"),
                        "section": c.get("section"),
                    },
                )
            )
        return docs

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        logger.info(
            (
                "HierarchicalRetriever request | bot_id=%s | top_clusters=%d | "
                "top_chunks=%d | source_filter=%s | doc_bm25=%s | social_bm25=%s | query=%s"
            ),
            self._bot_id,
            self._top_clusters,
            self._top_chunks,
            self._source_filter,
            int(self._enable_doc_bm25),
            int(self._enable_social_bm25),
            _preview_text(query),
        )
        source_filter = self._source_filter
        if isinstance(source_filter, (list, tuple, set)):
            source_filter = next(iter(source_filter), None)

        if source_filter == "website":
            bm25_chunks = search_bm25_chunks(
                query_text=query,
                client_id=self._client_id,
                bot_id=self._bot_id,
                top_k=max(self._top_chunks * 2, self._top_chunks),
                source_filter="website",
            )
            logger.info(
                "HierarchicalRetriever website-only BM25 | count=%d",
                len(bm25_chunks),
            )

            docs: List[Document] = []
            seen_refs: set[str] = set()
            for idx, c in enumerate(bm25_chunks[: self._top_chunks]):
                chunk_ref = c.get("chunk_ref") or f"website_{idx}"
                if chunk_ref in self._exclude_chunk_refs or chunk_ref in seen_refs:
                    continue
                seen_refs.add(chunk_ref)
                docs.append(
                    Document(
                        page_content=c.get("text", ""),
                        metadata={
                            "chunk_ref": chunk_ref,
                            "cluster": "website",
                            "chunk_index": idx,
                            "score": c.get("score", 0.0),
                            "topic": "website",
                            "pdf": c.get("pdf"),
                            "source_type": c.get("source_type"),
                            "source_url": c.get("source_url"),
                            "section": c.get("section")
                        }
                    )
                )

            # If BM25 misses/under-fills, fall back to semantic website index.
            if len(docs) < max(1, self._top_chunks):
                sem_docs = self._website_semantic_docs(
                    query,
                    top_k=max(self._top_chunks * 2, self._top_chunks),
                )
                logger.info(
                    "HierarchicalRetriever website-only semantic fallback | count=%d",
                    len(sem_docs),
                )
                for d in sem_docs:
                    ref = (d.metadata or {}).get("chunk_ref")
                    if not ref:
                        continue
                    if ref in seen_refs:
                        continue
                    seen_refs.add(ref)
                    docs.append(d)
                    if len(docs) >= self._top_chunks:
                        break
            logger.info(
                "HierarchicalRetriever website-only result | docs=%d",
                len(docs),
            )
            return docs[: self._top_chunks]

        # Stage 1 (all non-website-only prompts):
        # Search scraped website content first using BM25 only.
        website_stage_k = 0
        website_prefill_docs: List[Document] = []
        seen_refs: set[str] = set()
        if self._top_chunks > 0:
            website_stage_k = min(
                max(1, self._top_chunks // 2),
                max(1, self._top_chunks - 1),
            )
            website_chunks = search_bm25_chunks(
                query_text=query,
                client_id=self._client_id,
                bot_id=self._bot_id,
                top_k=max(self._top_chunks, website_stage_k * 3),
                source_filter="website",
            )
            logger.info(
                (
                    "HierarchicalRetriever stage1 website BM25 | requested=%d | "
                    "raw_count=%d | target_prefill=%d"
                ),
                max(self._top_chunks, website_stage_k * 3),
                len(website_chunks),
                website_stage_k,
            )
            for idx, c in enumerate(website_chunks):
                chunk_ref = c.get("chunk_ref") or f"website_{idx}"
                if chunk_ref in self._exclude_chunk_refs or chunk_ref in seen_refs:
                    continue
                seen_refs.add(chunk_ref)
                website_prefill_docs.append(
                    Document(
                        page_content=c.get("text", ""),
                        metadata={
                            "chunk_ref": chunk_ref,
                            "cluster": "website",
                            "chunk_index": idx,
                            "score": c.get("score", 0.0),
                            "topic": c.get("topic") or "website",
                            "pdf": c.get("pdf"),
                            "source_type": c.get("source_type"),
                            "source_url": c.get("source_url"),
                            "section": c.get("section"),
                        },
                    )
                )
                if len(website_prefill_docs) >= website_stage_k:
                    break

            if len(website_prefill_docs) < website_stage_k:
                sem_docs = self._website_semantic_docs(
                    query,
                    top_k=max(self._top_chunks, website_stage_k * 3),
                )
                logger.info(
                    (
                        "HierarchicalRetriever stage1 website semantic fallback | "
                        "count=%d | still_needed=%d"
                    ),
                    len(sem_docs),
                    max(0, website_stage_k - len(website_prefill_docs)),
                )
                for d in sem_docs:
                    ref = (d.metadata or {}).get("chunk_ref")
                    if not ref:
                        continue
                    if ref in self._exclude_chunk_refs or ref in seen_refs:
                        continue
                    seen_refs.add(ref)
                    website_prefill_docs.append(d)
                    if len(website_prefill_docs) >= website_stage_k:
                        break
            logger.info(
                "HierarchicalRetriever stage1 website selected | count=%d",
                len(website_prefill_docs),
            )

        # Stage 2:
        # Search manual uploaded docs with semantic + BM25 hybrid.
        query_top_chunks = self._top_chunks
        if self._source_filter:
            query_top_chunks = max(self._top_chunks * 4, self._top_chunks)

        query_embedding = self._query_embedding(query)

        chunks = query_hierarchical(
            query_embedding,
            self._client_id,
            self._bot_id,
            top_clusters=self._top_clusters,
            top_chunks=query_top_chunks,
            source_filter=self._source_filter,
            query_text=query,
            enable_bm25=self._enable_doc_bm25,
        )
        logger.info(
            "HierarchicalRetriever stage2 docs | count=%d | bm25_enabled=%s",
            len(chunks),
            int(self._enable_doc_bm25),
        )

        if self._source_filter:
            chunks = sorted(
                chunks,
                key=lambda c: c.get("score", 0.0),
                reverse=True
            )[: max(self._top_chunks, query_top_chunks)]

        docs: List[Document] = list(website_prefill_docs)
        for c in chunks:
            chunk_ref = c.get("chunk_ref") or f"{c.get('cluster')}_{c.get('chunk_index')}"
            if chunk_ref in self._exclude_chunk_refs or chunk_ref in seen_refs:
                continue
            seen_refs.add(chunk_ref)

            docs.append(
                Document(
                    page_content=c.get("text", ""),
                    metadata={
                        "chunk_ref": chunk_ref,
                        "cluster": c.get("cluster"),
                        "chunk_index": c.get("chunk_index"),
                        "score": c.get("score", 0.0),
                        "topic": c.get("topic"),
                        "pdf": c.get("pdf"),
                        "source_type": c.get("source_type"),
                        "source_url": c.get("source_url")
                    }
                )
            )
            if len(docs) >= self._top_chunks:
                break

        social_added = 0
        if (
            self._enable_social_bm25 and
            not self._source_filter and
            len(docs) < self._top_chunks
        ):
            needed = max(1, self._top_chunks - len(docs))
            social_chunks = search_bm25_chunks(
                query_text=query,
                client_id=self._client_id,
                bot_id=self._bot_id,
                top_k=max(needed * 3, needed),
                source_filter="social",
            )
            logger.info(
                "HierarchicalRetriever stage3 social BM25 | requested=%d | raw_count=%d",
                max(needed * 3, needed),
                len(social_chunks),
            )
            for idx, c in enumerate(social_chunks):
                chunk_ref = c.get("chunk_ref") or f"social_{idx}"
                if chunk_ref in self._exclude_chunk_refs or chunk_ref in seen_refs:
                    continue
                seen_refs.add(chunk_ref)
                docs.append(
                    Document(
                        page_content=c.get("text", ""),
                        metadata={
                            "chunk_ref": chunk_ref,
                            "cluster": c.get("cluster") or "social",
                            "chunk_index": c.get("chunk_index", idx),
                            "score": c.get("score", 0.0),
                            "topic": c.get("topic") or "social",
                            "pdf": c.get("pdf"),
                            "source_type": c.get("source_type") or "social",
                            "source_url": c.get("source_url"),
                        },
                    )
                )
                social_added += 1
                if len(docs) >= self._top_chunks:
                    break

        logger.info(
            "HierarchicalRetriever final result | docs=%d | website_prefill=%d | social_used=%s",
            len(docs),
            len(website_prefill_docs),
            int(social_added > 0),
        )
        return docs[: self._top_chunks]

    # Compatibility shim for older LangChain versions that don't expose
    # get_relevant_documents on BaseRetriever.
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query)
