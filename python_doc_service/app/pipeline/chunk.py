import spacy
import numpy as np
import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from app.pipeline.logger import get_logger

log = get_logger("chunk")

# --------------------------------------------------
# Models
# --------------------------------------------------



def _load_nlp():
    model_name = str(os.getenv("SPACY_MODEL", "en_core_web_sm")).strip() or "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except Exception as err:
        log.warning(
            "spaCy model unavailable; using lightweight sentencizer fallback | model=%s | err=%s",
            model_name,
            err,
        )
        nlp_fallback = spacy.blank("en")
        if "sentencizer" not in nlp_fallback.pipe_names:
            nlp_fallback.add_pipe("sentencizer")
        return nlp_fallback


nlp = _load_nlp()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# STEP 1: Extract all sentences globally
# --------------------------------------------------

def extract_sentences(sections):
    all_sentences = []
    section_ids = []

    for sec_id, section in enumerate(sections):
        for p in section["content"]:
            doc = nlp(p)
            for s in doc.sents:
                text = s.text.strip()
                if len(text.split()) >= 6:
                    all_sentences.append(text)
                    section_ids.append(sec_id)

    log.info(f"Global sentences extracted: {len(all_sentences)}")
    return all_sentences, section_ids


# --------------------------------------------------
# STEP 2: Detect boilerplate sentences (GLOBAL)
# --------------------------------------------------

def detect_boilerplate(sentences, section_ids):
    """
    Returns a set of sentence indices that are boilerplate
    """

    if len(sentences) < 20:
        log.info("Too few sentences for boilerplate detection")
        return set()

    # ---------- 1. TF-IDF frequency ----------
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_df=0.85,
        min_df=2
    )
    X = tfidf.fit_transform(sentences)
    tfidf_scores = np.asarray(X.sum(axis=1)).ravel()
    tfidf_cut = np.percentile(tfidf_scores, 75)

    tfidf_candidates = {
        i for i, v in enumerate(tfidf_scores) if v > tfidf_cut
    }

    # ---------- 2. Semantic centrality ----------
    embeddings = embedder.encode(
        sentences,
        normalize_embeddings=True,
        batch_size=32
    )
    centroid = embeddings.mean(axis=0)
    sims = cosine_similarity(
        embeddings,
        centroid.reshape(1, -1)
    ).ravel()
    semantic_cut = np.percentile(sims, 75)

    semantic_candidates = {
        i for i, s in enumerate(sims) if s > semantic_cut
    }

    # ---------- 3. Cross-section repetition ----------
    section_map = defaultdict(set)
    for i, sec in enumerate(section_ids):
        section_map[sentences[i]].add(sec)

    repetition_candidates = {
        i for i, s in enumerate(sentences)
        if len(section_map[s]) >= max(2, len(set(section_ids)) // 3)
    }

    boilerplate = (
        tfidf_candidates &
        semantic_candidates &
        repetition_candidates
    )

    log.info(
        f"Boilerplate detected: {len(boilerplate)} / {len(sentences)}"
    )
    return boilerplate


# --------------------------------------------------
# STEP 3: Robust adaptive chunk parameters
# --------------------------------------------------

def derive_chunk_params(stats):
    """
    Robust to different sentence_stats schemas
    """

    median = (
        stats.get("median_sent_len")
        or stats.get("median")
        or stats.get("median_sentence_length")
    )

    if median is None:
        raise ValueError(f"Unexpected sentence_stats schema: {stats}")

    return {
        "target_words": median * 4,   # ~3–5 sentences
        "min_words": median * 3,
        "max_sentences": 5
    }


# --------------------------------------------------
# STEP 4: Local semantic boilerplate filter
# --------------------------------------------------

def filter_semantic_boilerplate(sentences, threshold=0.92):
    """
    Removes sentences that dominate meaning locally
    (semantic boilerplate inside a chunk)
    """

    if len(sentences) <= 2:
        return sentences

    embeddings = embedder.encode(
        sentences,
        normalize_embeddings=True
    )
    centroid = embeddings.mean(axis=0)

    sims = cosine_similarity(
        embeddings,
        centroid.reshape(1, -1)
    ).ravel()

    filtered = [
        s for s, sim in zip(sentences, sims)
        if sim < threshold
    ]

    # Safety guard
    return filtered if len(filtered) >= 2 else sentences


# --------------------------------------------------
# STEP 5: Chunking with BOTH boilerplate filters
# --------------------------------------------------

def chunk_section(section, params, boilerplate_sentences):
    sentences = []
    for p in section["content"]:
        doc = nlp(p)
        sentences.extend([s.text.strip() for s in doc.sents])

    # ---- global boilerplate removal ----
    sentences = [
        s for s in sentences
        if s not in boilerplate_sentences and len(s.split()) >= 6
    ]

    if not sentences:
        return []

    log.debug(
        f"Section '{section['title']}' → {len(sentences)} usable sentences"
    )

    chunks = []
    i = 0

    while i < len(sentences):
        words = 0
        chunk_sents = []

        while (
            i < len(sentences)
            and words < params["target_words"]
            and len(chunk_sents) < params["max_sentences"]
        ):
            chunk_sents.append(sentences[i])
            words += len(sentences[i].split())
            i += 1

        # ---- local semantic boilerplate removal ----
        chunk_sents = filter_semantic_boilerplate(chunk_sents)

        if words < params["min_words"] and chunks:
            chunks[-1]["text"] += " " + " ".join(chunk_sents)
            continue

        chunks.append({
            "text": " ".join(chunk_sents),
            "section": section["title"],
            "pdf": section["meta"][0]["pdf"],
            "source_type": section["meta"][0].get("source_type"),
            "source_url": section["meta"][0].get("source_url")
        })

        # small overlap (avoid infinite loop when only 1 sentence)
        if len(chunk_sents) > 1:
            i = max(0, i - 1)

    log.info(
        f"Section '{section['title']}' → {len(chunks)} chunks"
    )

    return chunks
