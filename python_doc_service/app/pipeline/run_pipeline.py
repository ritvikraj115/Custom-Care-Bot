"""
run_pipeline.py

End-to-end document intelligence pipeline:
PDFs → blocks → sections → chunks → embeddings →
UMAP + HDBSCAN (auto-selected) → topic labels →
hierarchical vector index

COLAB-EQUIVALENT + BOILERPLATE-AWARE
"""

import numpy as np
import re
from collections import Counter
import time

from app.pipeline.extract import (
    extract_blocks_from_pdfs,
    mark_headings,
    sentence_stats,
    build_sections
)

from app.pipeline.chunk import (
    derive_chunk_params,
    chunk_section,
    extract_sentences,
    detect_boilerplate
)

from app.pipeline.embed import embed_chunks
from app.pipeline.clustering_pipeline import find_best_clustering
from app.pipeline.label import label_clusters
from app.pipeline.hierarchical_index import build_hierarchical_index, delete_bot_indexes
from app.pipeline.logger import get_logger
from app.pipeline.mlflow_tracking import (
    log_dict as mlflow_log_dict,
    log_metrics as mlflow_log_metrics,
    log_params as mlflow_log_params,
    start_bot_run as mlflow_start_bot_run,
)
from app.pipeline.vector_store import replace_website_chunks
from app.pipeline.elasticsearch_hybrid import index_bot_chunks as es_index_bot_chunks
from app.pipeline.elasticsearch_hybrid import append_bot_chunks as es_append_bot_chunks
from app.pipeline.elasticsearch_hybrid import delete_bot_chunks as es_delete_bot_chunks

log = get_logger("pipeline")


def _split_text_chunks(text, target_words=160, overlap=30):
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return chunks


WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_TEXT_RE = re.compile(r"[^a-z0-9\s]", re.IGNORECASE)

LOW_VALUE_WEBSITE_PATTERNS = (
    "cookie policy",
    "privacy policy",
    "terms of use",
    "all rights reserved",
    "follow us",
    "sign in",
    "log in",
    "logout",
    "dashboard",
)

NAV_WORDS = {
    "home",
    "about",
    "services",
    "contact",
    "blog",
    "login",
    "logout",
    "dashboard",
    "pricing",
    "menu",
}


def _words(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def _norm_line(text: str) -> str:
    text = URL_RE.sub(" ", str(text or ""))
    text = " ".join(text.split()).strip().lower()
    text = NON_TEXT_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def _split_website_sentences(text: str) -> list[str]:
    raw = str(text or "").replace("\x00", " ")
    raw = re.sub(r"[\r\n\t]+", "\n", raw)
    lines = []
    for part in raw.split("\n"):
        part = " ".join(part.split()).strip()
        if part:
            lines.append(part)
    if not lines:
        return []

    chunks = []
    for line in lines:
        split = re.split(r"(?<=[.!?])\s+|(?<=:)\s+", line)
        for s in split:
            s = " ".join(s.split()).strip(" -–•|")
            if s:
                chunks.append(s)
    return chunks


def _is_low_value_website_sentence(text: str) -> bool:
    sentence = " ".join(str(text or "").split()).strip()
    if not sentence:
        return True

    norm = _norm_line(sentence)
    tokens = norm.split()
    if not tokens:
        return True

    if len(tokens) <= 2 and all(tok in NAV_WORDS for tok in tokens):
        return True

    if len(tokens) <= 3:
        joined = " ".join(tokens)
        if joined in NAV_WORDS:
            return True

    lowered = sentence.lower()
    if any(pat in lowered for pat in LOW_VALUE_WEBSITE_PATTERNS):
        return True

    # Skip path-like/UI snippets with little lexical content.
    alpha_count = sum(ch.isalpha() for ch in sentence)
    if alpha_count < 8:
        return True

    return False


def _append_sentence_chunks(
    out: list[str],
    sentences: list[str],
    target_words: int = 110,
    min_words: int = 8,
    overlap_sentences: int = 1,
) -> None:
    if not sentences:
        return

    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sw = len(_words(sentence))
        if sw == 0:
            continue

        if current and current_words + sw > target_words:
            chunk_text = " ".join(current).strip()
            if len(_words(chunk_text)) >= min_words:
                out.append(chunk_text)

            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(overlap)
            current_words = sum(len(_words(x)) for x in current)

        current.append(sentence)
        current_words += sw

    if current:
        chunk_text = " ".join(current).strip()
        if len(_words(chunk_text)) >= min_words:
            out.append(chunk_text)


def _build_website_chunks(
    website_raw_chunks: list[dict],
    target_words: int = 110,
    min_words: int = 8,
) -> list[dict]:
    if not website_raw_chunks:
        return []

    # Global boilerplate detection over website sentences.
    global_counts: Counter[str] = Counter()
    prepared: list[tuple[dict, list[str]]] = []

    for row in website_raw_chunks:
        raw_sentences = _split_website_sentences(row.get("text", ""))
        cleaned_sentences: list[str] = []
        seen_local: set[str] = set()
        for sentence in raw_sentences:
            norm = _norm_line(sentence)
            if not norm or norm in seen_local:
                continue
            seen_local.add(norm)
            cleaned_sentences.append(sentence)
            global_counts[norm] += 1
        prepared.append((row, cleaned_sentences))

    chunk_count = max(1, len(website_raw_chunks))
    boilerplate_cut = max(3, int(round(chunk_count * 0.28)))
    boilerplate_norm = {
        s_norm
        for s_norm, count in global_counts.items()
        if count >= boilerplate_cut and len(s_norm.split()) <= 12
    }

    out: list[dict] = []
    seen_chunks: set[str] = set()

    for row, sentences in prepared:
        filtered = []
        for sentence in sentences:
            norm = _norm_line(sentence)
            if norm in boilerplate_norm:
                continue
            if _is_low_value_website_sentence(sentence):
                continue
            filtered.append(sentence)

        if not filtered:
            # Fallback: keep at least some informative text.
            filtered = [
                s
                for s in sentences
                if not _is_low_value_website_sentence(s)
            ][:8]

        text_windows: list[str] = []
        _append_sentence_chunks(
            text_windows,
            filtered,
            target_words=target_words,
            min_words=min_words,
            overlap_sentences=1,
        )

        for chunk_text in text_windows:
            norm_chunk = _norm_line(chunk_text)
            if not norm_chunk or norm_chunk in seen_chunks:
                continue
            seen_chunks.add(norm_chunk)
            out.append(
                {
                    **row,
                    "text": chunk_text,
                }
            )

    return out


def _sample_chunk_texts(chunks, limit=24, max_chars=260):
    out = []
    for chunk in chunks or []:
        text = " ".join(str(chunk.get("text", "")).split())
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        out.append(text)
        if len(out) >= int(limit):
            break
    return out


def run_pipeline(
    pdf_paths,
    bot_id,
    client_id,
    pdf_metadata=None,
    rebuild_mode: str = "full"
):
    rebuild_mode = str(rebuild_mode or "full").strip().lower()
    is_incremental = rebuild_mode == "incremental"

    log.info("===== PIPELINE START =====")
    log.info(
        "Client: %s, Bot: %s, Rebuild mode: %s",
        client_id,
        bot_id,
        rebuild_mode
    )

    # --------------------------------------------------
    # 1. STRUCTURE-AWARE EXTRACTION
    # --------------------------------------------------
    blocks = extract_blocks_from_pdfs(pdf_paths, pdf_metadata=pdf_metadata)
    blocks = mark_headings(blocks)

    stats = sentence_stats(blocks)
    sections = build_sections(blocks)

    log.info(f"Sections built: {len(sections)}")

    # --------------------------------------------------
    # 2. GLOBAL BOILERPLATE DETECTION (CRITICAL)
    # --------------------------------------------------
    all_sentences, section_ids = extract_sentences(sections)

    boilerplate_idxs = detect_boilerplate(
        all_sentences,
        section_ids
    )

    boilerplate_set = set(
        all_sentences[i] for i in boilerplate_idxs
    )

    log.info(
        f"Boilerplate sentences removed: "
        f"{len(boilerplate_set)}"
    )

    # --------------------------------------------------
    # 3. ADAPTIVE SEMANTIC CHUNKING
    # --------------------------------------------------
    chunk_params = derive_chunk_params(stats)

    doc_chunks = []
    website_raw_chunks = []

    for section in sections:
        meta0 = section.get("meta", [{}])[0] if section.get("meta") else {}
        sec_source_type = (meta0.get("source_type") or "upload").strip().lower()

        if sec_source_type == "website":
            # Website content performs better when we avoid doc-style aggressive
            # sentence pruning at this stage; apply dedicated cleaner/chunker later.
            sec_text = " ".join(str(p).strip() for p in section.get("content", []) if str(p).strip())
            sec_text = " ".join(sec_text.split()).strip()
            if sec_text:
                website_raw_chunks.append(
                    {
                        "text": sec_text,
                        "section": section.get("title"),
                        "pdf": meta0.get("pdf"),
                        "source_type": "website",
                        "source_url": meta0.get("source_url"),
                        "bot_id": bot_id,
                        "client_id": client_id,
                    }
                )
            continue

        section_chunks = chunk_section(
            section,
            chunk_params,
            boilerplate_set
        )

        for c in section_chunks:
            c["bot_id"] = bot_id
            c["client_id"] = client_id
            source_type = (c.get("source_type") or "upload").strip().lower()
            c["source_type"] = source_type
            if source_type == "website":
                website_raw_chunks.append(c)
            else:
                doc_chunks.append(c)

    if not doc_chunks and not website_raw_chunks:
        raise RuntimeError("No chunks produced from PDFs")

    log.info(
        "Chunks created | docs=%d | website=%d",
        len(doc_chunks),
        len(website_raw_chunks)
    )

    # --------------------------------------------------
    # 4. WEBSITE CHUNKS (NO CLUSTERING)
    # --------------------------------------------------
    website_chunks = []
    if website_raw_chunks:
        website_chunks = _build_website_chunks(
            website_raw_chunks,
            target_words=110,
            min_words=8,
        )

        if website_chunks:
            for idx, c in enumerate(website_chunks):
                c["chunk_ref"] = f"website_{idx}"

            avg_words = float(
                np.mean([len(_words(str(c.get("text", "")))) for c in website_chunks])
            ) if website_chunks else 0.0
            log.info(
                "Website chunk cleanup summary | raw=%d | cleaned=%d | avg_words=%.1f",
                len(website_raw_chunks),
                len(website_chunks),
                avg_words,
            )

            website_embeddings = embed_chunks(website_chunks)
            if is_incremental:
                # Incremental document-only ingestion should not overwrite
                # existing website chunks unless website content is present.
                replace_website_chunks(bot_id, website_chunks, website_embeddings)
            else:
                replace_website_chunks(bot_id, website_chunks, website_embeddings)
        else:
            if not is_incremental:
                log.warning("Website chunks empty after split; clearing website index.")
                replace_website_chunks(bot_id, [], [])
    elif not is_incremental:
        replace_website_chunks(bot_id, [], [])

    labels = np.array([], dtype=int)
    best_params = {}
    embeddings = None

    # --------------------------------------------------
    # 5. DOCUMENT CLUSTERING (UMAP + HDBSCAN)
    # --------------------------------------------------
    if doc_chunks:
        embeddings = embed_chunks(doc_chunks)

        clustering_started = time.time()
        labels, best_params, clustering_results = find_best_clustering(embeddings)

        for i, lbl in enumerate(labels):
            doc_chunks[i]["cluster"] = int(lbl)

        if is_incremental:
            # Keep historical cluster IDs stable and avoid collisions while
            # appending newly indexed document batches.
            suffix = int(time.time())
            for i, lbl in enumerate(labels):
                doc_chunks[i]["cluster"] = f"inc_{suffix}_{int(lbl)}"

        # --------------------------------------------------
        # 6. TOPIC LABELING
        # --------------------------------------------------
        cluster_topics = label_clusters(doc_chunks)

        for c in doc_chunks:
            c["topic"] = cluster_topics.get(c["cluster"], "noise")

        cluster_counts = {}
        for c in doc_chunks:
            cluster_counts[c["cluster"]] = cluster_counts.get(c["cluster"], 0) + 1

        log.info(f"Cluster sizes: {cluster_counts}")
        log.info(f"Best clustering params: {best_params}")

        with mlflow_start_bot_run(
            component="clustering",
            client_id=client_id,
            bot_id=bot_id,
            run_name=f"clustering-{client_id}-{bot_id}",
            extra_tags={"rebuild_mode": rebuild_mode},
        ) as mlflow_tracker:
            mlflow_log_params(
                mlflow_tracker,
                {
                    "rebuild_mode": rebuild_mode,
                    "doc_chunks": len(doc_chunks),
                    "embedding_dim": int(embeddings.shape[1]) if hasattr(embeddings, "shape") and len(embeddings.shape) > 1 else 0,
                    "configs_evaluated": len(clustering_results),
                    "best_n_components": best_params.get("n_components"),
                    "best_n_neighbors": best_params.get("n_neighbors"),
                    "best_min_cluster_size": best_params.get("min_cluster_size"),
                    "best_min_samples": best_params.get("min_samples"),
                },
            )
            mlflow_log_metrics(
                mlflow_tracker,
                {
                    "cluster_count": float(len(set(labels)) - (1 if -1 in labels else 0)),
                    "noise_ratio": float(np.mean(labels == -1)),
                    "clustering_score": float(best_params.get("score", 0.0) or 0.0),
                    "clustering_seconds": float(time.time() - clustering_started),
                },
            )
            mlflow_log_dict(
                mlflow_tracker,
                {
                    "best_params": best_params,
                    "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
                },
                "clustering_summary.json",
            )

        # --------------------------------------------------
        # 7. HIERARCHICAL VECTOR INDEX
        # --------------------------------------------------
        for c, emb in zip(doc_chunks, embeddings):
            c["embedding"] = emb

        build_hierarchical_index(
            doc_chunks,
            embeddings,
            reset=not is_incremental
        )
    else:
        if not is_incremental:
            delete_bot_indexes(client_id, bot_id)
            log.info("No document chunks; cleared hierarchical indexes.")

    es_rows = list(doc_chunks)
    if website_chunks:
        for idx, row in enumerate(website_chunks):
            row.setdefault("cluster", "website")
            row.setdefault("topic", "website")
            row.setdefault("chunk_index", int(idx))
            row.setdefault("chunk_ref", f"website_{idx}")
            es_rows.append(row)

    try:
        if es_rows:
            if is_incremental:
                es_append_bot_chunks(client_id, bot_id, es_rows)
            else:
                es_index_bot_chunks(client_id, bot_id, es_rows)
        else:
            if not is_incremental:
                es_delete_bot_chunks(client_id, bot_id)
    except Exception as err:
        log.warning("Elasticsearch chunk sync failed | bot_id=%s | err=%s", bot_id, err)


    # --------------------------------------------------
    # 8. QUALITY CHECKS (DEBUG MODE)
    # --------------------------------------------------
    import os
    run_quality = os.getenv("RUN_QUALITY_CHECKS", "false").strip().lower() in {
        "1", "true", "yes", "y"
    }
    if run_quality and doc_chunks and embeddings is not None:
        from app.pipeline.quality.run_quality import run_quality_checks
        run_quality_checks(
            doc_chunks,
            embeddings,
            client_id,
            bot_id
        )
    elif run_quality:
        log.info("Quality checks skipped (no document chunks)")
    else:
        log.info("Quality checks skipped (RUN_QUALITY_CHECKS=false)")

    # --------------------------------------------------
    # 9. SUMMARY
    # --------------------------------------------------
    doc_cluster_count = 0
    noise_ratio = 0.0
    if doc_chunks and labels.size:
        doc_cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = float(np.mean(labels == -1))

    summary = {
        "bot_id": bot_id,
        "client_id": client_id,
        "pdfs_processed": len(pdf_paths),
        "doc_chunks": len(doc_chunks),
        "website_chunks": len(website_chunks),
        "total_chunks": len(doc_chunks) + len(website_chunks),
        "clusters": doc_cluster_count,
        "noise_ratio": noise_ratio,
        "best_params": best_params,
        "rebuild_mode": rebuild_mode,
        # Lightweight context snippets for downstream autocomplete bootstrapping.
        "doc_samples": _sample_chunk_texts(doc_chunks, limit=18),
        "website_samples": _sample_chunk_texts(website_chunks, limit=12),
    }

    log.info("===== PIPELINE END =====")
    return summary
