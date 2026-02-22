import faiss
import numpy as np
import os

EMBED_DIM = 384
EXPERIENCE_MIN_SIMILARITY = min(
    0.95,
    max(
        0.55,
        float(os.getenv("EXPERIENCE_MIN_SIMILARITY", "0.72"))
    )
)

# ==================================================
# 1) DOCUMENT CHUNK VECTOR STORE (EXISTING RAG)
# ==================================================

chunk_index = faiss.IndexFlatIP(EMBED_DIM)
chunk_metadata = []

# ==================================================
# 1b) WEBSITE CHUNK VECTOR STORE (NO CLUSTERING)
# ==================================================

website_index = faiss.IndexFlatIP(EMBED_DIM)
website_metadata = []


def _rebuild_index(index, metadata, keep_fn):
    new_index = faiss.IndexFlatIP(EMBED_DIM)
    new_meta = []

    if index.ntotal == 0:
        return new_index, new_meta

    for i, meta in enumerate(metadata):
        if not keep_fn(meta):
            continue
        try:
            vec = index.reconstruct(i)
        except Exception:
            # If reconstruct fails, skip this entry
            continue
        new_index.add(np.array([vec]).astype("float32"))
        new_meta.append(meta)

    return new_index, new_meta


def store_chunks(chunks, embeddings):
    """
    Store document chunks for RAG
    """
    vectors = np.array(embeddings).astype("float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    chunk_index.add(vectors)

    for c in chunks:
        chunk_metadata.append({
            "bot_id": c["bot_id"],
            "client_id": c["client_id"],
            "topic": c.get("topic"),
            "text": c["text"]
        })


def search_chunks(query_embedding, bot_id, top_k=5):
    """
    Search document chunks for RAG
    """
    if chunk_index.ntotal == 0:
        return []

    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    scores, ids = chunk_index.search(
        np.array([query_embedding]).astype("float32"),
        top_k
    )

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        meta = chunk_metadata[idx]
        if meta["bot_id"] != bot_id:
            continue

        results.append({
            "text": meta["text"],
            "topic": meta["topic"],
            "score": float(score)
        })

    return results


def replace_website_chunks(bot_id, chunks, embeddings):
    """
    Replace all website chunks for a bot (no clustering).
    """
    global website_index, website_metadata
    keep_fn = lambda meta: meta.get("bot_id") != bot_id
    website_index, website_metadata = _rebuild_index(
        website_index,
        website_metadata,
        keep_fn
    )

    if not chunks:
        return True

    vectors = np.array(embeddings).astype("float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    website_index.add(vectors)

    for c in chunks:
        website_metadata.append({
            "bot_id": c["bot_id"],
            "client_id": c["client_id"],
            "text": c["text"],
            "section": c.get("section"),
            "pdf": c.get("pdf"),
            "chunk_ref": c.get("chunk_ref"),
            "source_type": c.get("source_type"),
            "source_url": c.get("source_url")
        })

    return True


def search_website_chunks(query_embedding, bot_id, top_k=5):
    """
    Search website chunks only (no clustering).
    """
    if website_index.ntotal == 0:
        return []

    query_embedding = np.array(query_embedding).astype("float32")
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    k = min(top_k, website_index.ntotal)
    scores, ids = website_index.search(
        np.array([query_embedding]),
        k
    )

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        meta = website_metadata[idx]
        if meta["bot_id"] != bot_id:
            continue
        results.append({
            "text": meta["text"],
            "section": meta.get("section"),
            "pdf": meta.get("pdf"),
            "chunk_ref": meta.get("chunk_ref"),
            "source_type": meta.get("source_type"),
            "source_url": meta.get("source_url"),
            "score": float(score)
        })

    return results


def delete_chunks_for_bot(bot_id):
    """
    Remove all chunk vectors for a bot.
    """
    global chunk_index, chunk_metadata
    keep_fn = lambda meta: meta.get("bot_id") != bot_id
    chunk_index, chunk_metadata = _rebuild_index(
        chunk_index,
        chunk_metadata,
        keep_fn
    )
    return True


def delete_website_chunks_for_bot(bot_id):
    """
    Remove all website chunk vectors for a bot.
    """
    global website_index, website_metadata
    keep_fn = lambda meta: meta.get("bot_id") != bot_id
    website_index, website_metadata = _rebuild_index(
        website_index,
        website_metadata,
        keep_fn
    )
    return True


# ==================================================
# 2) EXPERIENCE VECTOR STORE
# ==================================================

experience_index = faiss.IndexFlatIP(EMBED_DIM)
experience_metadata = []
ALPHA = 0.15


def _find_semantic_match(
    query_embedding,
    bot_id,
    top_k=5,
    min_similarity=EXPERIENCE_MIN_SIMILARITY
):
    """
    Find best semantic match for a bot (by similarity only)
    """
    if experience_index.ntotal == 0:
        return None

    query_embedding = np.array(query_embedding).astype("float32")
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    k = min(top_k, experience_index.ntotal)
    scores, ids = experience_index.search(
        np.array([query_embedding]),
        k
    )

    best = None
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        meta = experience_metadata[idx]
        if meta["bot_id"] != bot_id:
            continue
        if score < min_similarity:
            continue
        if best is None or score > best["similarity"]:
            best = {"idx": idx, "similarity": float(score)}

    return best


def store_experience(embedding, meta):
    """
    Store one experience (question embedding)
    """
    embedding = np.array(embedding).astype("float32")
    embedding = embedding / np.linalg.norm(embedding)

    match = _find_semantic_match(
        embedding,
        meta["bot_id"]
    )

    if match is not None:
        existing = experience_metadata[match["idx"]]

        # Owner answer is pinned; do not replace with non-owner
        if existing.get("owner_answer") and not meta.get("owner_answer"):
            return

        existing["experience_id"] = meta["experience_id"]
        existing["semantic_id"] = meta.get(
            "semantic_id",
            existing.get("semantic_id")
        )
        existing["feedback_score"] = meta.get(
            "feedback_score",
            existing.get("feedback_score", 0)
        )
        existing["avg_chunk_similarity"] = meta.get(
            "avg_chunk_similarity",
            existing.get("avg_chunk_similarity", 0.0)
        )
        existing["negative_count"] = meta.get(
            "negative_count",
            existing.get("negative_count", 0)
        )
        existing["owner_answer"] = meta.get(
            "owner_answer",
            existing.get("owner_answer", False)
        )
        return

    experience_index.add(np.array([embedding]))

    # REQUIRED FIELDS IN meta:
    # experience_id, bot_id, feedback_score, avg_chunk_similarity
    experience_metadata.append({
        "experience_id": meta["experience_id"],
        "bot_id": meta["bot_id"],
        "semantic_id": meta.get("semantic_id"),
        "feedback_score": meta.get("feedback_score", 0),
        "avg_chunk_similarity": meta.get("avg_chunk_similarity", 0.0),
        "negative_count": meta.get("negative_count", 0),
        "owner_answer": meta.get("owner_answer", False)
    })


def search_experience(
    query_embedding,
    bot_id,
    top_k=5,
    min_similarity=EXPERIENCE_MIN_SIMILARITY
):
    """
    Search semantically similar past experiences (question memory)
    """

    if experience_index.ntotal == 0:
        return None

    query_embedding = np.array(query_embedding).astype("float32")
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    k = min(top_k, experience_index.ntotal)
    scores, ids = experience_index.search(
        np.array([query_embedding]),
        k
    )

    # DEBUG: inspect candidates
    print("Experience candidates:")
    for s, i in zip(scores[0], ids[0]):
        print(i, s)

    best = None
    best_owner = None

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        meta = experience_metadata[idx]

        # 1) bot isolation
        if meta["bot_id"] != bot_id:
            continue

        # 2) semantic threshold
        if score < min_similarity:
            continue

        feedback_score = meta.get("feedback_score", 0) or 0
        rank_score = float(score) + ALPHA * float(feedback_score)

        candidate = {
            "experience_id": meta["experience_id"],
            "semantic_id": meta.get("semantic_id"),
            "similarity": float(score),
            "rank_score": rank_score,
            "feedback_score": feedback_score,
            "avg_chunk_similarity": meta.get(
                "avg_chunk_similarity",
                0.0
            ),
            "negative_count": meta.get("negative_count", 0),
            "owner_answer": meta.get("owner_answer", False)
        }

        if meta.get("owner_answer"):
            if (
                best_owner is None
                or candidate["similarity"] > best_owner["similarity"]
            ):
                best_owner = candidate
            continue

        if (
            best is None
            or candidate["rank_score"] > best["rank_score"]
        ):
            best = candidate

    return best_owner or best


def update_experience_feedback(
    experience_id=None,
    delta=0,
    semantic_id=None,
    bot_id=None,
    feedback_score=None,
    negative_count=None
):
    """
    Update feedback metadata for an existing experience vector
    """
    updated = False

    def _apply(meta):
        nonlocal updated
        current_score = meta.get("feedback_score", 0) or 0
        current_negative = meta.get("negative_count", 0) or 0
        if semantic_id:
            meta["semantic_id"] = semantic_id
        if feedback_score is not None:
            meta["feedback_score"] = float(feedback_score)
        else:
            meta["feedback_score"] = current_score + float(delta or 0)
        if negative_count is not None:
            meta["negative_count"] = int(negative_count)
        elif float(delta or 0) < 0:
            meta["negative_count"] = current_negative + 1
        updated = True

    if semantic_id and bot_id:
        semantic_key = str(semantic_id)
        bot_key = str(bot_id)
        for meta in experience_metadata:
            if str(meta.get("bot_id")) != bot_key:
                continue
            if str(meta.get("semantic_id")) != semantic_key:
                continue
            _apply(meta)
        if updated:
            return True

    if experience_id:
        exp_key = str(experience_id)
        for meta in experience_metadata:
            if str(meta.get("experience_id")) != exp_key:
                continue
            _apply(meta)
            return True

    return updated


def delete_experiences_for_bot(bot_id):
    """
    Remove all experience vectors for a bot.
    """
    global experience_index, experience_metadata
    keep_fn = lambda meta: meta.get("bot_id") != bot_id
    experience_index, experience_metadata = _rebuild_index(
        experience_index,
        experience_metadata,
        keep_fn
    )
    return True


def increment_negative_for_bot(bot_id):
    """
    Increment negative count for all experiences of a bot
    (used when semantic group is negatively reinforced)
    """
    for meta in experience_metadata:
        if meta["bot_id"] == bot_id:
            meta["negative_count"] = meta.get("negative_count", 0) + 1


def get_negative_count_for_bot(bot_id):
    return max(
        (meta.get("negative_count", 0)
         for meta in experience_metadata
         if meta["bot_id"] == bot_id),
        default=0
    )
