import faiss
import numpy as np
import hashlib
import json
import os
import shutil
from collections import defaultdict
from app.pipeline.logger import get_logger
from app.pipeline.model_monitoring import record_hdbscan_query_event
from app.pipeline.storage import ensure_dir, index_bot_dir
from app.pipeline.elasticsearch_hybrid import (
    HYBRID_BM25_WEIGHT,
    HYBRID_SEMANTIC_WEIGHT,
    is_enabled as es_is_enabled,
    search_bm25_chunks,
)

log = get_logger("index")

EMBED_DIM = 384

# ============================================================
# GLOBAL IN-MEMORY STORES
# ============================================================

# (client_id, bot_id) -> FAISS index of cluster centroids
CLUSTER_INDEX = {}

# (client_id, bot_id) -> list of cluster_ids aligned with centroid index
CLUSTER_META = {}

# (client_id, bot_id, cluster_id) -> FAISS index of chunk embeddings
CHUNK_INDEX = {}

# (client_id, bot_id, cluster_id) -> list of chunk metadata
CHUNK_META = {}


def _chunk_ref(cluster_id, chunk_index: int) -> str:
    return f"{cluster_id}_{int(chunk_index)}"


def _normalize_semantic(score: float) -> float:
    val = float(score)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _persist_chunk_file_id(cluster_id) -> str:
    raw = str(cluster_id).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _chunks_dir(client_id: str, bot_id: str) -> str:
    return os.path.join(index_bot_dir(client_id, bot_id), "chunks")


def _delete_persisted_indexes(client_id: str, bot_id: str) -> None:
    path = index_bot_dir(client_id, bot_id)
    if os.path.exists(path):
        shutil.rmtree(path)


def save_bot_indexes(client_id: str, bot_id: str) -> bool:
    cluster_key = (client_id, bot_id)
    if cluster_key not in CLUSTER_INDEX:
        return False

    root = index_bot_dir(client_id, bot_id)
    ensure_dir(root)
    chunks_root = _chunks_dir(client_id, bot_id)
    if os.path.exists(chunks_root):
        shutil.rmtree(chunks_root)
    ensure_dir(chunks_root)

    cluster_index = CLUSTER_INDEX[cluster_key]
    cluster_meta = CLUSTER_META.get(cluster_key, [])
    faiss.write_index(
        cluster_index,
        os.path.join(root, "clusters.faiss")
    )
    with open(
        os.path.join(root, "clusters.meta.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(cluster_meta, f, ensure_ascii=True)

    partitions_saved = 0
    for key, idx in CHUNK_INDEX.items():
        key_client_id, key_bot_id, cluster_id = key
        if key_client_id != client_id or key_bot_id != bot_id:
            continue

        file_id = _persist_chunk_file_id(cluster_id)
        index_file = os.path.join(chunks_root, f"{file_id}.faiss")
        meta_file = os.path.join(chunks_root, f"{file_id}.meta.json")

        faiss.write_index(idx, index_file)
        meta_payload = {
            "cluster_id": cluster_id,
            "chunks": CHUNK_META.get(key, [])
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=True)
        partitions_saved += 1

    log.info(
        "Persisted bot indexes | client_id=%s | bot_id=%s | partitions=%d",
        client_id,
        bot_id,
        partitions_saved
    )
    return True


def delete_bot_indexes(client_id: str, bot_id: str) -> int:
    """
    Delete all in-memory indexes for a specific (client, bot).
    Returns number of chunk index partitions removed.
    """
    cluster_key = (client_id, bot_id)
    if cluster_key in CLUSTER_INDEX:
        CLUSTER_INDEX.pop(cluster_key, None)
    if cluster_key in CLUSTER_META:
        CLUSTER_META.pop(cluster_key, None)

    removed = 0
    for key in list(CHUNK_INDEX.keys()):
        if key[0] == client_id and key[1] == bot_id:
            CHUNK_INDEX.pop(key, None)
            CHUNK_META.pop(key, None)
            removed += 1

    log.info(
        "Deleted bot indexes | client_id=%s | bot_id=%s | chunk_partitions=%d",
        client_id,
        bot_id,
        removed
    )
    _delete_persisted_indexes(client_id, bot_id)
    return removed

# ============================================================
# BUILD INDEXES (CALLED AFTER CLUSTERING)
# ============================================================
def build_hierarchical_index(chunks, embeddings, reset=True):
    """
    Build hierarchical FAISS indexes:
    client → bot → cluster → chunks
    """

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings length mismatch")

    # --------------------------------------------------
    # RESET indexes for affected (client, bot)
    # --------------------------------------------------
    affected_keys = set(
        (c["client_id"], c["bot_id"]) for c in chunks
    )

    if reset:

        for key in affected_keys:
            CLUSTER_INDEX[key] = faiss.IndexFlatIP(EMBED_DIM)
            CLUSTER_META[key] = []

        # clear chunk-level indexes
        for c in chunks:
            CHUNK_INDEX.pop(
                (c["client_id"], c["bot_id"], c["cluster"]),
                None
            )
            CHUNK_META.pop(
                (c["client_id"], c["bot_id"], c["cluster"]),
                None
            )

    # --------------------------------------------------
    # GROUP embeddings by cluster (SAFE)
    # --------------------------------------------------
    cluster_vectors = defaultdict(list)
    cluster_items = defaultdict(list)

    for idx, (c, emb) in enumerate(zip(chunks, embeddings)):
        emb = emb / np.linalg.norm(emb)

        key = (c["client_id"], c["bot_id"], c["cluster"])
        cluster_vectors[key].append(emb)
        cluster_items[key].append((c, emb))

    # --------------------------------------------------
    # BUILD CLUSTER-LEVEL INDEX (ROUTING)
    # --------------------------------------------------
    for (client_id, bot_id, cluster_id), vecs in cluster_vectors.items():
        cluster_key = (client_id, bot_id)
        if cluster_key not in CLUSTER_INDEX:
            CLUSTER_INDEX[cluster_key] = faiss.IndexFlatIP(EMBED_DIM)
            CLUSTER_META[cluster_key] = []

        centroid = np.mean(vecs, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        CLUSTER_INDEX[cluster_key].add(
            np.asarray([centroid], dtype="float32")
        )
        CLUSTER_META[cluster_key].append(cluster_id)

    # --------------------------------------------------
    # BUILD CHUNK-LEVEL INDEX (RETRIEVAL)
    # --------------------------------------------------
    for (client_id, bot_id, cluster_id), items in cluster_items.items():
        chunk_key = (client_id, bot_id, cluster_id)

        idx = faiss.IndexFlatIP(EMBED_DIM)
        meta = []

        for local_idx, (c, emb) in enumerate(items):
            idx.add(np.asarray([emb], dtype="float32"))
            c["chunk_index"] = int(local_idx)
            c["chunk_ref"] = _chunk_ref(cluster_id, local_idx)
            meta.append({
                "text": c["text"],
                "topic": c.get("topic"),
                "cluster": c["cluster"],
                "chunk_index": int(local_idx),
                "chunk_ref": _chunk_ref(cluster_id, local_idx),
                "pdf": c.get("pdf"),
                "source_type": c.get("source_type"),
                "source_url": c.get("source_url")
            })

        CHUNK_INDEX[chunk_key] = idx
        CHUNK_META[chunk_key] = meta

    log.info(
        f"Cluster index rebuilt for {len(CLUSTER_INDEX)} (client, bot) pairs"
    )
    log.info(
        f"Chunk indexes rebuilt: {len(CHUNK_INDEX)}"
    )

    for client_id, bot_id in affected_keys:
        save_bot_indexes(client_id, bot_id)

# ============================================================
# QUERY (FAST, TENANT-SAFE, HIERARCHICAL)
# ============================================================

def query_hierarchical(
    query_embedding,
    client_id,
    bot_id,
    top_clusters=2,
    top_chunks=5,
    source_filter=None,
    query_text: str | None = None,
    enable_bm25: bool = True,
):
    """
    Hierarchical retrieval:
    1. Route query to top-N clusters
    2. Search chunks ONLY inside those clusters
    """

    cluster_key = (client_id, bot_id)

    allowed_sources = None
    if source_filter:
        if isinstance(source_filter, (list, tuple, set)):
            allowed_sources = {str(s) for s in source_filter}
        else:
            allowed_sources = {str(source_filter)}

    def _bm25_only_fallback(reason: str) -> list[dict]:
        if not query_text or not es_is_enabled() or not enable_bm25:
            record_hdbscan_query_event(
                client_id=client_id,
                bot_id=bot_id,
                final_hits=0,
                semantic_hits=0,
                candidate_clusters=0,
                reason=reason,
            )
            return []

        bm25_results = search_bm25_chunks(
            query_text=query_text,
            client_id=client_id,
            bot_id=bot_id,
            top_k=max(6, int(top_chunks) * max(1, int(top_clusters))),
            source_filter=allowed_sources,
            cluster_filter=None,
        )
        if not bm25_results:
            record_hdbscan_query_event(
                client_id=client_id,
                bot_id=bot_id,
                final_hits=0,
                semantic_hits=0,
                candidate_clusters=0,
                reason=f"{reason}_bm25_empty",
            )
            return []

        out: list[dict] = []
        for row in bm25_results:
            out.append(
                {
                    "text": row.get("text", ""),
                    "topic": row.get("topic"),
                    "cluster": row.get("cluster"),
                    "chunk_index": int(row.get("chunk_index", 0) or 0),
                    "chunk_ref": row.get("chunk_ref"),
                    "score": float(row.get("score", 0.0) or 0.0),
                    "semantic_score": 0.0,
                    "source_type": row.get("source_type"),
                    "source_url": row.get("source_url"),
                    "pdf": row.get("pdf"),
                }
            )
        record_hdbscan_query_event(
            client_id=client_id,
            bot_id=bot_id,
            final_hits=len(out),
            semantic_hits=0,
            candidate_clusters=0,
            reason=f"{reason}_bm25_fallback",
        )
        log.debug(
            "Retrieval summary | client_id=%s | bot_id=%s | semantic_hits=0 | bm25_hits=%d | mode=bm25_only_fallback | reason=%s",
            client_id,
            bot_id,
            len(out),
            reason,
        )
        return out

    if cluster_key not in CLUSTER_INDEX:
        return _bm25_only_fallback("no_cluster_index")

    # normalize query for cosine similarity
    denom = float(np.linalg.norm(query_embedding))
    if denom <= 0.0:
        record_hdbscan_query_event(
            client_id=client_id,
            bot_id=bot_id,
            final_hits=0,
            semantic_hits=0,
            candidate_clusters=0,
            reason="invalid_query_embedding",
        )
        return []
    query_embedding = query_embedding / denom

    # --------------------------------------------------
    # STEP 1: CLUSTER ROUTING
    # --------------------------------------------------
    cluster_index = CLUSTER_INDEX[cluster_key]
    cluster_meta = CLUSTER_META[cluster_key]

    k = min(top_clusters, cluster_index.ntotal)
    if k <= 0:
        return _bm25_only_fallback("empty_cluster_index")

    D, I = cluster_index.search(
        np.asarray([query_embedding], dtype="float32"),
        k
    )

    candidate_clusters = [
        cluster_meta[i]
        for i in I[0]
        if 0 <= i < len(cluster_meta)
    ]

    # --------------------------------------------------
    # STEP 2: CHUNK RETRIEVAL INSIDE CLUSTERS
    # --------------------------------------------------
    semantic_results = []

    for cluster_id in candidate_clusters:
        chunk_key = (client_id, bot_id, cluster_id)

        if chunk_key not in CHUNK_INDEX:
            continue

        idx = CHUNK_INDEX[chunk_key]
        meta = CHUNK_META[chunk_key]

        if idx.ntotal == 0:
            continue

        k2 = min(top_chunks, idx.ntotal)

        D2, I2 = idx.search(
            np.asarray([query_embedding], dtype="float32"),
            k2
        )

        for score, i in zip(D2[0], I2[0]):
            if i < 0 or i >= len(meta):
                continue
            if allowed_sources is not None:
                source_type = meta[i].get("source_type")
                if source_type not in allowed_sources:
                    continue
            semantic_results.append({
                "text": meta[i]["text"],
                "topic": meta[i].get("topic"),
                "cluster": cluster_id,
                "chunk_index": int(i),
                "chunk_ref": meta[i].get("chunk_ref") or _chunk_ref(cluster_id, int(i)),
                "score": float(score),
                "semantic_score": float(score),
                "source_type": meta[i].get("source_type"),
                "source_url": meta[i].get("source_url"),
                "pdf": meta[i].get("pdf")
            })

    if not semantic_results and (not query_text or not es_is_enabled() or not enable_bm25):
        log.debug(
            "Retrieval summary | client_id=%s | bot_id=%s | semantic_hits=0 | mode=semantic_only_no_results",
            client_id,
            bot_id,
        )
        record_hdbscan_query_event(
            client_id=client_id,
            bot_id=bot_id,
            final_hits=0,
            semantic_hits=0,
            candidate_clusters=len(candidate_clusters),
            reason="semantic_only_no_results",
        )
        return []

    if not query_text or not es_is_enabled() or not enable_bm25:
        semantic_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        log.debug(
            "Retrieval summary | client_id=%s | bot_id=%s | semantic_hits=%d | mode=semantic_only | bm25_enabled=%s",
            client_id,
            bot_id,
            len(semantic_results),
            int(bool(enable_bm25)),
        )
        record_hdbscan_query_event(
            client_id=client_id,
            bot_id=bot_id,
            final_hits=len(semantic_results),
            semantic_hits=len(semantic_results),
            candidate_clusters=len(candidate_clusters),
            reason="semantic_only",
        )
        return semantic_results

    bm25_results = search_bm25_chunks(
        query_text=query_text,
        client_id=client_id,
        bot_id=bot_id,
        top_k=max(6, int(top_chunks) * max(2, int(top_clusters))),
        source_filter=allowed_sources,
        cluster_filter=candidate_clusters,
    )

    if not bm25_results:
        if not semantic_results:
            log.debug(
                "Hybrid retrieval summary | client_id=%s | bot_id=%s | semantic_hits=0 | bm25_hits=0 | final_hits=0",
                client_id,
                bot_id,
            )
            record_hdbscan_query_event(
                client_id=client_id,
                bot_id=bot_id,
                final_hits=0,
                semantic_hits=0,
                candidate_clusters=len(candidate_clusters),
                reason="hybrid_no_results",
            )
            return []
        semantic_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        log.debug(
            "Hybrid retrieval summary | client_id=%s | bot_id=%s | semantic_hits=%d | bm25_hits=0 | final_hits=%d",
            client_id,
            bot_id,
            len(semantic_results),
            len(semantic_results),
        )
        record_hdbscan_query_event(
            client_id=client_id,
            bot_id=bot_id,
            final_hits=len(semantic_results),
            semantic_hits=len(semantic_results),
            candidate_clusters=len(candidate_clusters),
            reason="hybrid_bm25_empty",
        )
        return semantic_results

    # Weighted fusion of semantic (FAISS cosine) and lexical (ES BM25).
    bm25_max = max(float(row.get("score", 0.0) or 0.0) for row in bm25_results)
    bm25_max = max(bm25_max, 1e-9)
    merged: dict[str, dict] = {}

    for row in semantic_results:
        ref = str(row.get("chunk_ref") or _chunk_ref(row.get("cluster"), row.get("chunk_index", 0)))
        merged[ref] = {
            **row,
            "semantic_norm": _normalize_semantic(float(row.get("semantic_score", row.get("score", 0.0)) or 0.0)),
            "bm25_norm": 0.0,
            "bm25_score": 0.0,
        }

    for row in bm25_results:
        ref = str(row.get("chunk_ref") or _chunk_ref(row.get("cluster"), row.get("chunk_index", 0)))
        bm25_raw = float(row.get("score", 0.0) or 0.0)
        bm25_norm = max(0.0, min(1.0, bm25_raw / bm25_max))

        existing = merged.get(ref)
        if existing is None:
            merged[ref] = {
                "text": row.get("text", ""),
                "topic": row.get("topic"),
                "cluster": row.get("cluster"),
                "chunk_index": int(row.get("chunk_index", 0) or 0),
                "chunk_ref": ref,
                "source_type": row.get("source_type"),
                "source_url": row.get("source_url"),
                "pdf": row.get("pdf"),
                "semantic_score": 0.0,
                "semantic_norm": 0.0,
                "bm25_score": bm25_raw,
                "bm25_norm": bm25_norm,
            }
        else:
            existing["bm25_score"] = bm25_raw
            existing["bm25_norm"] = max(float(existing.get("bm25_norm", 0.0)), bm25_norm)

    fused: list[dict] = []
    for row in merged.values():
        sem_norm = float(row.get("semantic_norm", 0.0) or 0.0)
        bm_norm = float(row.get("bm25_norm", 0.0) or 0.0)

        has_sem = sem_norm > 0.0
        has_bm25 = bm_norm > 0.0

        if has_sem and has_bm25:
            score = (HYBRID_SEMANTIC_WEIGHT * sem_norm) + (HYBRID_BM25_WEIGHT * bm_norm)
        elif has_sem:
            # Preserve baseline behavior if lexical signal is missing.
            score = sem_norm
        else:
            # Lexical-only candidates are allowed but weaker by default.
            score = HYBRID_BM25_WEIGHT * bm_norm

        row["score"] = float(score)
        fused.append(row)

    fused.sort(
        key=lambda item: (
            float(item.get("score", 0.0) or 0.0),
            float(item.get("semantic_norm", 0.0) or 0.0),
            float(item.get("bm25_norm", 0.0) or 0.0),
        ),
        reverse=True,
    )

    result_cap = max(1, len(candidate_clusters)) * max(1, int(top_chunks))
    final = fused[:result_cap]
    log.debug(
        (
            "Hybrid retrieval summary | client_id=%s | bot_id=%s | clusters=%d | "
            "semantic_hits=%d | bm25_hits=%d | final_hits=%d | weights=semantic:%.2f bm25:%.2f"
        ),
        client_id,
        bot_id,
        len(candidate_clusters),
        len(semantic_results),
        len(bm25_results),
        len(final),
        HYBRID_SEMANTIC_WEIGHT,
        HYBRID_BM25_WEIGHT,
    )
    record_hdbscan_query_event(
        client_id=client_id,
        bot_id=bot_id,
        final_hits=len(final),
        semantic_hits=len(semantic_results),
        candidate_clusters=len(candidate_clusters),
        reason="hybrid_fused",
    )
    return final
