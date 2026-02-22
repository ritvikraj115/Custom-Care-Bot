"""
Bot-specific autocomplete training pipeline.

Keeps the same model family as your original script:
- SentencePiece unigram tokenizer
- Hybrid token + character encoder transformer decoder

Adds production behavior:
- Bot-specific bootstrap from bot content questions
- Per-bot question logging
- Retrain trigger every 25 new user questions
- Recency-weighted training samples
- Data augmentation for small bot datasets
- Variable-length phrase suggestions with confidence
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import string
import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from app.pipeline.storage import BASE_DIR, ensure_dir
from app.pipeline.logger import get_logger
from app.pipeline.elasticsearch_hybrid import (
    AUTOCOMPLETE_ES_WEIGHT,
    AUTOCOMPLETE_MODEL_WEIGHT,
    index_question_event as es_index_question_event,
    replace_bot_questions as es_replace_bot_questions,
    search_question_suggestions as es_search_question_suggestions,
)
from app.pipeline.dvc_auto import track_autocomplete_training as dvc_track_autocomplete_training
from app.pipeline.airflow_trigger import (
    airflow_autotrigger_enabled,
    trigger_autocomplete_retrain_dag,
)
from app.pipeline.mlflow_tracking import (
    log_artifact as mlflow_log_artifact,
    log_metrics as mlflow_log_metrics,
    log_params as mlflow_log_params,
    set_tags as mlflow_set_tags,
    start_bot_run as mlflow_start_bot_run,
)
from app.pipeline.model_monitoring import record_autocomplete_query_event


# ===============================
# Constants
# ===============================

BASE_ACTIONS = [
    "refund",
    "track",
    "cancel",
    "update",
    "change",
    "check",
    "apply",
    "reset",
    "download",
    "activate",
    "verify",
    "modify",
    "report",
    "request",
    "remove",
    "upgrade",
]

BASE_OBJECTS = [
    "my order",
    "my subscription",
    "my account",
    "my password",
    "delivery address",
    "payment method",
    "billing details",
    "shipping address",
    "order status",
    "refund status",
    "invoice copy",
    "email address",
    "phone number",
    "transaction history",
    "security settings",
]

BASE_EXTRAS = [
    "as soon as possible",
    "for my recent purchase",
    "because it was delayed",
    "due to incorrect billing",
    "for last month's order",
    "using my registered email",
    "before renewal date",
    "immediately",
    "today",
    "for international shipment",
]

POLITE_PREFIXES = [
    "please",
    "can you",
    "could you",
    "help me",
    "i need to",
    "how do i",
]

COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}

# Complete short words that are valid as independent tokens.
SHORT_WORD_ALLOWLIST = {
    "a",
    "an",
    "as",
    "at",
    "be",
    "by",
    "do",
    "go",
    "he",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "of",
    "on",
    "or",
    "so",
    "to",
    "up",
    "us",
    "we",
    "the",
}

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")

AUTOCOMPLETE_ROOT = os.path.join(BASE_DIR, "autocomplete")

log = get_logger("autocomplete")


# ===============================
# Utilities
# ===============================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(raw: str) -> str:
    return SAFE_ID_RE.sub("_", str(raw or "unknown")).strip("_") or "unknown"


def _normalize_text(text: str) -> str:
    return " ".join(WORD_RE.findall((text or "").lower()))


def _count_words(text: str) -> int:
    return len(WORD_RE.findall((text or "").lower()))


def _add_typo(word: str, rng: random.Random) -> str:
    if len(word) <= 3:
        return word
    i = rng.randint(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def _augment_question(question: str, rng: random.Random, max_aug: int = 3) -> list[str]:
    base = _normalize_text(question)
    if not base:
        return []

    words = base.split()
    variants: set[str] = set()

    for prefix in POLITE_PREFIXES:
        variants.add(f"{prefix} {base}")

    if words:
        typo_words = list(words)
        idx = rng.randrange(len(words))
        typo_words[idx] = _add_typo(typo_words[idx], rng)
        variants.add(" ".join(typo_words))

    out = [v for v in variants if v and v != base]
    rng.shuffle(out)
    return out[: max(0, int(max_aug))]


def _build_seed_queries(size: int = 300, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    templates = [
        "{a} {o}",
        "{a} {o} {x}",
        "please {a} {o}",
        "can you {a} {o}",
        "how do i {a} {o}",
        "i need to {a} {o}",
        "what is the status of {o}",
        "why is my {o} pending",
        "where can i {a} {o}",
        "help me {a} {o} {x}",
        "can i {a} {o} {x}",
        "how can i {a} {o}",
        "is it possible to {a} {o}",
        "guide me to {a} {o}",
        "i want to {a} {o} now",
        "need support to {a} {o}",
        "i cannot {a} {o}",
        "issue with {o}",
        "problem with {o}",
        "question about {o}",
    ]

    rows: list[str] = []
    for _ in range(max(600, int(size))):
        t = rng.choice(templates)
        q = t.format(
            a=rng.choice(BASE_ACTIONS),
            o=rng.choice(BASE_OBJECTS),
            x=rng.choice(BASE_EXTRAS),
        )
        rows.append(_normalize_text(q))
        if rng.random() < 0.25:
            rows.append(_normalize_text(f"please {q}"))
        if rng.random() < 0.18:
            rows.append(_normalize_text(f"{q} urgently"))

    rows = [r for r in rows if r]
    return list(dict.fromkeys(rows))


def _extract_context_terms(texts: list[str], limit: int = 30) -> list[str]:
    freq: dict[str, int] = {}
    for raw in texts:
        norm = _normalize_text(raw)
        if not norm:
            continue
        for token in norm.split():
            if len(token) < 4:
                continue
            if token in COMMON_STOPWORDS:
                continue
            freq[token] = freq.get(token, 0) + 1

    ranked = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    return [term for term, _ in ranked[: max(5, int(limit))]]


# ===============================
# Model Architecture (same family as original)
# ===============================

def build_char_encoder(max_char_len: int, char_vocab_size: int, char_emb_dim: int = 32, out_dim: int = 128):
    char_inputs = keras.Input(shape=(max_char_len,))
    x = layers.Embedding(char_vocab_size, char_emb_dim)(char_inputs)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(out_dim)(x)
    return keras.Model(char_inputs, x)


def build_hybrid_model(
    vocab_size: int,
    max_seq_len: int,
    max_char_len: int,
    char_vocab_size: int,
    d_model: int = 128,
    num_heads: int = 4,
    dff: int = 256,
    num_layers: int = 2,
    dropout_rate: float = 0.1,
):
    token_inputs = layers.Input(shape=(max_seq_len,))
    char_inputs = layers.Input(shape=(max_seq_len, max_char_len))

    token_emb = layers.Embedding(vocab_size, d_model)(token_inputs)

    char_encoder = build_char_encoder(max_char_len, char_vocab_size, out_dim=d_model)
    char_emb = layers.TimeDistributed(char_encoder)(char_inputs)

    x = token_emb + char_emb

    pos = tf.range(start=0, limit=max_seq_len, delta=1)
    pos_emb = layers.Embedding(max_seq_len, d_model)(pos)
    x = x + pos_emb

    mask = tf.linalg.band_part(tf.ones((max_seq_len, max_seq_len)), -1, 0)

    for _ in range(num_layers):
        ln1 = layers.LayerNormalization()(x)
        attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )(ln1, ln1, attention_mask=mask)
        x = x + attn

        ln2 = layers.LayerNormalization()(x)
        ffn = layers.Dense(dff, activation="gelu")(ln2)
        ffn = layers.Dense(d_model)(ffn)
        x = x + ffn

    x = layers.LayerNormalization()(x)
    x = x[:, -1, :]
    outputs = layers.Dense(vocab_size)(x)
    return keras.Model([token_inputs, char_inputs], outputs)


def perplexity(y_true, y_pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return tf.exp(tf.reduce_mean(loss))


# ===============================
# Manager
# ===============================

class BotAutocompleteManager:
    def __init__(
        self,
        retrain_every: int = 25,
        min_training_questions: int = 25,
        recency_boost: float = 3.6,
        max_questions_per_bot: int = 4000,
        max_seq_len: int = 24,
        max_char_len: int = 15,
        vocab_size: int = 240,
    ) -> None:
        self.retrain_every = max(1, int(retrain_every))
        self.min_training_questions = max(1, int(min_training_questions))
        env_recency_boost = os.getenv("AUTOCOMPLETE_RECENCY_BOOST")
        if env_recency_boost is not None:
            try:
                recency_boost = float(env_recency_boost)
            except Exception:
                pass
        self.recency_boost = max(0.0, float(recency_boost))
        self.max_questions_per_bot = max(200, int(max_questions_per_bot))

        self.recency_power = max(1.2, float(os.getenv("AUTOCOMPLETE_RECENCY_POWER", "2.4")))
        self.seed_source_weight = min(
            1.0,
            max(0.1, float(os.getenv("AUTOCOMPLETE_SEED_WEIGHT", "0.45"))),
        )
        self.recent_user_window = max(
            10,
            int(os.getenv("AUTOCOMPLETE_RECENT_USER_WINDOW", "40")),
        )
        self.recent_user_multiplier = max(
            1.0,
            float(os.getenv("AUTOCOMPLETE_RECENT_USER_MULTIPLIER", "1.55")),
        )
        self.ultra_recent_window = max(
            5,
            int(os.getenv("AUTOCOMPLETE_ULTRA_RECENT_WINDOW", "12")),
        )
        self.ultra_recent_multiplier = max(
            1.0,
            float(os.getenv("AUTOCOMPLETE_ULTRA_RECENT_MULTIPLIER", "1.18")),
        )
        self.semantic_dedupe_threshold = min(
            0.98,
            max(
                0.6,
                float(os.getenv("AUTOCOMPLETE_SEMANTIC_DEDUPE_THRESHOLD", "0.82")),
            ),
        )
        self.semantic_dedupe_window = max(
            40,
            int(os.getenv("AUTOCOMPLETE_SEMANTIC_DEDUPE_WINDOW", "180")),
        )

        self.max_seq_len = max(8, int(max_seq_len))
        self.max_char_len = max(8, int(max_char_len))
        self.vocab_size = max(120, int(vocab_size))

        char_vocab = list(string.ascii_lowercase + string.digits + " _'")
        self.char_to_idx = {c: i + 1 for i, c in enumerate(char_vocab)}
        self.char_vocab_size = len(self.char_to_idx) + 1

        ensure_dir(AUTOCOMPLETE_ROOT)

        self._sp_cache: dict[str, spm.SentencePieceProcessor] = {}
        self._model_cache: dict[str, tuple[float, keras.Model]] = {}
        self._char_token_cache: dict[str, list[int]] = {}
        self._piece_text_cache: dict[tuple[int, int], str] = {}
        self._suggest_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._suggest_cache_ttl_ms = 30 * 1000
        self._suggest_cache_max = 6000

        self._bot_locks: dict[tuple[str, str], threading.Lock] = {}
        self._inflight: set[tuple[str, str]] = set()
        self._queued_airflow: dict[tuple[str, str], float] = {}
        self._global_lock = threading.Lock()
        self._semantic_embedder = None
        self._airflow_autotrigger = airflow_autotrigger_enabled()
        self._airflow_queue_ttl_sec = max(
            60,
            int(os.getenv("AIRFLOW_QUEUE_TTL_SEC", "900")),
        )

        log.info(
            (
                "Autocomplete config | retrain_every=%d | min_train=%d | "
                "recency_boost=%.2f | recency_power=%.2f | seed_weight=%.2f | "
                "recent_window=%d | recent_mult=%.2f | ultra_window=%d | ultra_mult=%.2f | "
                "semantic_dedupe=%.2f/%d | hybrid_weights=model:%.2f es:%.2f"
            ),
            self.retrain_every,
            self.min_training_questions,
            self.recency_boost,
            self.recency_power,
            self.seed_source_weight,
            self.recent_user_window,
            self.recent_user_multiplier,
            self.ultra_recent_window,
            self.ultra_recent_multiplier,
            self.semantic_dedupe_threshold,
            self.semantic_dedupe_window,
            AUTOCOMPLETE_MODEL_WEIGHT,
            AUTOCOMPLETE_ES_WEIGHT,
        )
        log.info(
            "Autocomplete startup ready | base_training_on_startup=false | strategy=bot_specific_bootstrap"
        )
        log.info(
            "Autocomplete orchestration | airflow_autotrigger=%s | queue_ttl_sec=%d",
            bool(self._airflow_autotrigger),
            int(self._airflow_queue_ttl_sec),
        )

    # -------------------------------
    # Paths / state
    # -------------------------------

    def _bot_key(self, client_id: str, bot_id: str) -> tuple[str, str]:
        return _safe_id(client_id), _safe_id(bot_id)

    def _bot_lock(self, client_id: str, bot_id: str) -> threading.Lock:
        key = self._bot_key(client_id, bot_id)
        with self._global_lock:
            if key not in self._bot_locks:
                self._bot_locks[key] = threading.Lock()
            return self._bot_locks[key]

    def _bot_dir(self, client_id: str, bot_id: str) -> str:
        safe_client, safe_bot = self._bot_key(client_id, bot_id)
        return os.path.join(AUTOCOMPLETE_ROOT, f"client_{safe_client}", f"bot_{safe_bot}")

    def _prune_airflow_queue(self, now_ts: float) -> None:
        stale_before = float(now_ts) - float(self._airflow_queue_ttl_sec)
        stale_keys = [k for k, ts in self._queued_airflow.items() if float(ts) <= stale_before]
        for key in stale_keys:
            self._queued_airflow.pop(key, None)

    def _questions_path(self, client_id: str, bot_id: str) -> str:
        return os.path.join(self._bot_dir(client_id, bot_id), "questions.jsonl")

    def _state_path(self, client_id: str, bot_id: str) -> str:
        return os.path.join(self._bot_dir(client_id, bot_id), "state.json")

    def _model_path(self, client_id: str, bot_id: str) -> str:
        return os.path.join(self._bot_dir(client_id, bot_id), "model.keras")

    def _weights_path(self, client_id: str, bot_id: str) -> str:
        return os.path.join(self._bot_dir(client_id, bot_id), "model.weights.h5")

    def _tokenizer_prefix(self, client_id: str, bot_id: str) -> str:
        return os.path.join(self._bot_dir(client_id, bot_id), "tokenizer")

    def _tokenizer_model_path(self, client_id: str, bot_id: str) -> str:
        return self._tokenizer_prefix(client_id, bot_id) + ".model"

    def _default_state(self) -> dict[str, Any]:
        return {
            # user-entered questions only
            "total_questions": 0,
            # synthetic/doc-grounded seed questions
            "total_seed_questions": 0,
            "total_corpus_questions": 0,
            "bootstrap_completed": False,

            # backward-compatible aliases
            "last_trained_count": 0,
            "last_trained_user_count": 0,
            "last_trained_corpus_count": 0,

            "model_version": 0,
            "is_training": False,
            "last_trained_at": None,
            "last_training_seconds": None,
            "last_error": None,
            "updated_at": _utc_now_iso(),
        }

    def _load_json(self, path: str, default: dict[str, Any]) -> dict[str, Any]:
        if not os.path.exists(path):
            return dict(default)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return dict(default)

    def _save_json(self, path: str, payload: dict[str, Any]) -> None:
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _load_state(self, client_id: str, bot_id: str) -> dict[str, Any]:
        state = self._load_json(self._state_path(client_id, bot_id), self._default_state())
        for k, v in self._default_state().items():
            state.setdefault(k, v)
        state["last_trained_user_count"] = int(
            state.get("last_trained_user_count", state.get("last_trained_count", 0)) or 0
        )
        state["last_trained_count"] = int(state.get("last_trained_user_count", 0) or 0)
        state["total_questions"] = int(state.get("total_questions", 0) or 0)
        state["total_seed_questions"] = int(state.get("total_seed_questions", 0) or 0)
        state["total_corpus_questions"] = int(
            state.get(
                "total_corpus_questions",
                int(state["total_questions"]) + int(state["total_seed_questions"]),
            )
            or 0
        )
        state["last_trained_corpus_count"] = int(
            state.get("last_trained_corpus_count", state.get("last_trained_user_count", 0)) or 0
        )
        return state

    def _save_state(self, client_id: str, bot_id: str, state: dict[str, Any]) -> None:
        state["updated_at"] = _utc_now_iso()
        self._save_json(self._state_path(client_id, bot_id), state)

    def _load_questions(self, client_id: str, bot_id: str) -> list[dict[str, Any]]:
        path = self._questions_path(client_id, bot_id)
        if not os.path.exists(path):
            return []

        rows: list[dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    q = _normalize_text(str(obj.get("question", "")))
                    if not q:
                        continue
                    ts = float(obj.get("ts", time.time()))
                    source = str(obj.get("source") or "").strip().lower()
                    if source not in {"user", "seed"}:
                        source = "seed" if bool(obj.get("is_seed")) else "user"
                    rows.append(
                        {
                            "question": q,
                            "ts": ts,
                            "source": source,
                        }
                    )
        except Exception:
            return []

        rows.sort(key=lambda item: item["ts"])
        if len(rows) > self.max_questions_per_bot:
            rows = rows[-self.max_questions_per_bot :]
        return rows

    def _write_questions(self, client_id: str, bot_id: str, rows: list[dict[str, Any]]) -> None:
        path = self._questions_path(client_id, bot_id)
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                payload = {
                    "question": _normalize_text(str(row.get("question", ""))),
                    "ts": float(row.get("ts", time.time())),
                    "source": "seed" if str(row.get("source", "user")) == "seed" else "user",
                }
                if payload["question"]:
                    f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _append_question(
        self,
        client_id: str,
        bot_id: str,
        question: str,
        source: str = "user",
        ts: float | None = None,
    ) -> None:
        path = self._questions_path(client_id, bot_id)
        ensure_dir(os.path.dirname(path))
        payload = {
            "question": _normalize_text(question),
            "ts": float(ts if ts is not None else time.time()),
            "source": "seed" if str(source).strip().lower() == "seed" else "user",
        }
        if not payload["question"]:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _count_question_sources(self, questions: list[dict[str, Any]]) -> tuple[int, int]:
        user_count = 0
        seed_count = 0
        for row in questions:
            if str(row.get("source", "user")) == "seed":
                seed_count += 1
            else:
                user_count += 1
        return user_count, seed_count

    def set_semantic_embedder(self, embedder: Any) -> None:
        self._semantic_embedder = embedder

    def _find_semantic_duplicate_user_question(
        self,
        question: str,
        rows: list[dict[str, Any]],
    ) -> tuple[str, float] | None:
        normalized = _normalize_text(question)
        if not normalized:
            return None

        user_rows = [
            _normalize_text(str(row.get("question", "")))
            for row in rows
            if str(row.get("source", "user")) != "seed"
        ]
        if not user_rows:
            return None

        # Fast exact guard before embedding pass.
        for q in reversed(user_rows):
            if q == normalized:
                return q, 1.0

        embedder = self._semantic_embedder
        if embedder is None:
            return None

        dedupe_candidates = list(dict.fromkeys(reversed(user_rows)))
        dedupe_candidates = dedupe_candidates[: self.semantic_dedupe_window]
        dedupe_candidates = [q for q in dedupe_candidates if q and q != normalized]
        if not dedupe_candidates:
            return None

        try:
            query_vec = embedder.encode(
                normalized,
                normalize_embeddings=True,
            )
            candidate_vecs = embedder.encode(
                dedupe_candidates,
                normalize_embeddings=True,
            )
        except Exception:
            return None

        if getattr(candidate_vecs, "ndim", 0) != 2:
            return None

        sims = np.dot(candidate_vecs, query_vec)
        if sims.size == 0:
            return None

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score < self.semantic_dedupe_threshold:
            return None

        return dedupe_candidates[best_idx], best_score

    # -------------------------------
    # Bot tokenizer artifacts
    # -------------------------------

    def _load_sp_model(self, model_path: str) -> spm.SentencePieceProcessor | None:
        if not model_path or not os.path.isfile(model_path):
            return None

        with self._global_lock:
            cached = self._sp_cache.get(model_path)
            if cached is not None:
                return cached

        sp = spm.SentencePieceProcessor()
        try:
            sp.load(model_path)
        except Exception:
            return None

        with self._global_lock:
            self._sp_cache[model_path] = sp
        return sp

    def _load_bot_sp(self, client_id: str, bot_id: str) -> spm.SentencePieceProcessor | None:
        return self._load_sp_model(self._tokenizer_model_path(client_id, bot_id))

    def _ensure_bot_tokenizer(
        self,
        client_id: str,
        bot_id: str,
        questions: list[dict[str, Any]],
    ) -> spm.SentencePieceProcessor | None:
        model_path = self._tokenizer_model_path(client_id, bot_id)
        cached = self._load_sp_model(model_path)
        if cached is not None:
            return cached

        rows: list[str] = []
        rng = random.Random(131)
        for row in questions:
            q = _normalize_text(str(row.get("question", "")))
            if not q:
                continue
            rows.append(q)
            for aug in _augment_question(q, rng, max_aug=1):
                rows.append(aug)

        if len(rows) < 120:
            rows.extend(_build_seed_queries(size=300, seed=23))

        rows = [_normalize_text(r) for r in rows if _normalize_text(r)]
        rows = list(dict.fromkeys(rows))
        if not rows:
            return None

        prefix = self._tokenizer_prefix(client_id, bot_id)
        ensure_dir(os.path.dirname(prefix))
        self._train_sentencepiece(rows, prefix)

        with self._global_lock:
            self._sp_cache.pop(model_path, None)

        return self._load_sp_model(model_path)

    def _train_sentencepiece(self, corpus_rows: list[str], model_prefix: str) -> None:
        corpus_rows = [_normalize_text(r) for r in corpus_rows if _normalize_text(r)]
        if not corpus_rows:
            raise ValueError("No corpus rows for tokenizer training")

        corpus_path = model_prefix + "_corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            for row in corpus_rows:
                f.write(row + "\n")

        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            hard_vocab_limit=False,
        )

    # -------------------------------
    # Training data prep
    # -------------------------------

    def _encode_char_token(self, token: str) -> list[int]:
        token = (token or "")[: self.max_char_len]
        cached = self._char_token_cache.get(token)
        if cached is not None:
            return list(cached)

        arr = [self.char_to_idx.get(c, 0) for c in token]
        if len(arr) < self.max_char_len:
            arr.extend([0] * (self.max_char_len - len(arr)))

        self._char_token_cache[token] = list(arr)
        return arr

    def _piece_text(self, sp: spm.SentencePieceProcessor, token_id: int) -> str:
        token_id = int(token_id)
        cache_key = (id(sp), token_id)
        cached = self._piece_text_cache.get(cache_key)
        if cached is not None:
            return cached

        # SentencePiece uses U+2581 as word boundary; strip for char path.
        piece = sp.id_to_piece(token_id).replace("\u2581", "")
        self._piece_text_cache[cache_key] = piece
        return piece

    def _is_valid_generation_piece(self, sp: spm.SentencePieceProcessor, token_id: int) -> bool:
        token_id = int(token_id)
        if token_id == int(sp.unk_id()):
            return False

        raw_piece = sp.id_to_piece(token_id)
        if not raw_piece:
            return False

        # U+2581 indicates word boundary in SentencePiece.
        is_word_start = raw_piece.startswith("\u2581")
        clean_piece = raw_piece.replace("\u2581", "").lower()
        if not clean_piece:
            return False

        if clean_piece.isdigit():
            return True

        piece_len = len(clean_piece)
        if piece_len >= 4:
            return True

        # For short pieces (<4), only allow if they are complete standalone words.
        if is_word_start and clean_piece in SHORT_WORD_ALLOWLIST:
            return True

        return False

    def _build_training_arrays(
        self,
        sp: spm.SentencePieceProcessor,
        weighted_rows: list[tuple[str, float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_tokens: list[np.ndarray] = []
        X_chars: list[np.ndarray] = []
        y: list[int] = []
        sample_weights: list[float] = []

        for text, row_weight in weighted_rows:
            text = _normalize_text(text)
            if not text:
                continue

            seq = sp.encode(text, out_type=int)
            if len(seq) < 2:
                continue

            for i in range(1, len(seq)):
                prefix = seq[max(0, i - self.max_seq_len) : i]
                target = int(seq[i])

                token_arr = np.zeros((self.max_seq_len,), dtype=np.int32)
                token_arr[-len(prefix) :] = np.array(prefix, dtype=np.int32)
                X_tokens.append(token_arr)

                char_arr = np.zeros((self.max_seq_len, self.max_char_len), dtype=np.int32)
                start = self.max_seq_len - len(prefix)
                for j, token_id in enumerate(prefix):
                    piece = self._piece_text(sp, int(token_id))
                    char_arr[start + j] = np.array(self._encode_char_token(piece), dtype=np.int32)
                X_chars.append(char_arr)

                y.append(target)
                sample_weights.append(float(max(0.05, row_weight)))

        if not y:
            return (
                np.zeros((0, self.max_seq_len), dtype=np.int32),
                np.zeros((0, self.max_seq_len, self.max_char_len), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
            )

        return (
            np.array(X_tokens, dtype=np.int32),
            np.array(X_chars, dtype=np.int32),
            np.array(y, dtype=np.int32),
            np.array(sample_weights, dtype=np.float32),
        )

    def _build_compiled_model(self, vocab_size: int) -> keras.Model:
        model = build_hybrid_model(
            vocab_size=vocab_size,
            max_seq_len=self.max_seq_len,
            max_char_len=self.max_char_len,
            char_vocab_size=self.char_vocab_size,
            d_model=128,
            num_heads=4,
            dff=256,
            num_layers=2,
            dropout_rate=0.1,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(3e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[perplexity],
        )
        return model

    # -------------------------------
    # Recency weighting + training
    # -------------------------------

    def _build_recency_weighted_rows(self, questions: list[dict[str, Any]]) -> list[tuple[str, float]]:
        if not questions:
            return []

        n = len(questions)
        rng = random.Random(101)
        recent_user_window = min(max(12, self.recent_user_window), max(12, n))
        ultra_recent_window = min(max(6, self.ultra_recent_window), max(6, n))

        merged: dict[str, float] = {}

        for idx, row in enumerate(questions):
            q = _normalize_text(row.get("question", ""))
            if not q:
                continue
            source = str(row.get("source", "user")).strip().lower()
            source_weight = 1.0 if source != "seed" else self.seed_source_weight

            if n == 1:
                ratio = 1.0
            else:
                ratio = float(idx) / float(n - 1)

            weight = (1.0 + (self.recency_boost * (ratio ** self.recency_power))) * source_weight
            if source != "seed" and idx >= max(0, n - recent_user_window):
                # Stronger focus to recent user asks.
                weight *= self.recent_user_multiplier
            if source != "seed" and idx >= max(0, n - ultra_recent_window):
                # Additional bump for the very latest user asks.
                weight *= self.ultra_recent_multiplier

            merged[q] = max(merged.get(q, 0.0), weight)

            for aug in _augment_question(q, rng, max_aug=2):
                aug_scale = 0.42 if source != "seed" else 0.3
                merged[aug] = max(merged.get(aug, 0.0), weight * aug_scale)

        return [(q, w) for q, w in merged.items()]

    def _train_bot_model(self, client_id: str, bot_id: str) -> None:
        start = time.time()
        lock = self._bot_lock(client_id, bot_id)

        with lock:
            state = self._load_state(client_id, bot_id)
            questions = self._load_questions(client_id, bot_id)
            user_count, seed_count = self._count_question_sources(questions)
            corpus_count = len(questions)
            state["total_questions"] = int(user_count)
            state["total_seed_questions"] = int(seed_count)
            state["total_corpus_questions"] = int(corpus_count)
            self._save_state(client_id, bot_id, state)

        with mlflow_start_bot_run(
            component="autocomplete",
            client_id=client_id,
            bot_id=bot_id,
            run_name=f"autocomplete-{client_id}-{bot_id}",
        ) as mlflow_tracker:
            mlflow_log_params(
                mlflow_tracker,
                {
                    "retrain_every": self.retrain_every,
                    "min_training_questions": self.min_training_questions,
                    "max_seq_len": self.max_seq_len,
                    "max_char_len": self.max_char_len,
                    "char_vocab_size": self.char_vocab_size,
                },
            )
            mlflow_log_metrics(
                mlflow_tracker,
                {
                    "user_questions": float(user_count),
                    "seed_questions": float(seed_count),
                    "corpus_questions": float(corpus_count),
                },
            )

            if len(questions) < self.min_training_questions:
                with lock:
                    state = self._load_state(client_id, bot_id)
                    state["is_training"] = False
                    state["last_error"] = f"requires_at_least_{self.min_training_questions}_questions"
                    self._save_state(client_id, bot_id, state)
                mlflow_set_tags(
                    mlflow_tracker,
                    {
                        "status": "skipped",
                        "skip_reason": f"requires_at_least_{self.min_training_questions}_questions",
                    },
                )
                return

            sp = self._ensure_bot_tokenizer(client_id, bot_id, questions)
            if sp is None:
                with lock:
                    state = self._load_state(client_id, bot_id)
                    state["is_training"] = False
                    state["last_error"] = "tokenizer_unavailable"
                    self._save_state(client_id, bot_id, state)
                mlflow_set_tags(
                    mlflow_tracker,
                    {"status": "failed", "skip_reason": "tokenizer_unavailable"},
                )
                return

            weighted_rows = self._build_recency_weighted_rows(questions)
            X_tokens, X_chars, y, sample_weights = self._build_training_arrays(sp, weighted_rows)

            weights_only = [float(w) for _, w in weighted_rows]
            if weights_only:
                weight_min = min(weights_only)
                weight_avg = float(sum(weights_only) / max(1, len(weights_only)))
                weight_max = max(weights_only)
                log.info(
                    (
                        "Autocomplete train summary | client_id=%s | bot_id=%s | user_q=%d | seed_q=%d | corpus_q=%d | "
                        "weighted_rows=%d | weight[min=%.3f avg=%.3f max=%.3f]"
                    ),
                    client_id,
                    bot_id,
                    int(user_count),
                    int(seed_count),
                    int(corpus_count),
                    len(weighted_rows),
                    weight_min,
                    weight_avg,
                    weight_max,
                )
                mlflow_log_metrics(
                    mlflow_tracker,
                    {
                        "weighted_rows": float(len(weighted_rows)),
                        "weight_min": float(weight_min),
                        "weight_avg": float(weight_avg),
                        "weight_max": float(weight_max),
                    },
                )
            else:
                log.info(
                    "Autocomplete train summary | client_id=%s | bot_id=%s | user_q=%d | seed_q=%d | corpus_q=%d | weighted_rows=0",
                    client_id,
                    bot_id,
                    int(user_count),
                    int(seed_count),
                    int(corpus_count),
                )
                mlflow_log_metrics(
                    mlflow_tracker,
                    {"weighted_rows": 0.0},
                )

            if len(y) == 0:
                with lock:
                    state = self._load_state(client_id, bot_id)
                    state["is_training"] = False
                    state["last_error"] = "empty_training_rows"
                    self._save_state(client_id, bot_id, state)
                mlflow_set_tags(
                    mlflow_tracker,
                    {"status": "skipped", "skip_reason": "empty_training_rows"},
                )
                return

            mlflow_log_metrics(
                mlflow_tracker,
                {"train_samples": float(len(y))},
            )

            model = self._build_compiled_model(sp.get_piece_size())
            bot_weights_path = self._weights_path(client_id, bot_id)
            bot_model_path = self._model_path(client_id, bot_id)
            if os.path.exists(bot_weights_path):
                try:
                    model.load_weights(bot_weights_path)
                except Exception:
                    pass
            elif os.path.exists(bot_model_path):
                try:
                    prev = keras.models.load_model(bot_model_path, compile=False)
                    model.set_weights(prev.get_weights())
                except Exception:
                    pass

            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_perplexity",
                patience=4,
                restore_best_weights=True,
                mode="min",
            )

            history = model.fit(
                [X_tokens, X_chars],
                y,
                sample_weight=sample_weights,
                validation_split=0.15,
                epochs=28,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0,
            )

            ensure_dir(os.path.dirname(bot_model_path))
            model.save(bot_model_path)
            model.save_weights(bot_weights_path)

            training_seconds = round(time.time() - start, 3)
            with lock:
                state = self._load_state(client_id, bot_id)
                state["last_trained_count"] = int(user_count)
                state["last_trained_user_count"] = int(user_count)
                state["last_trained_corpus_count"] = int(corpus_count)
                state["total_questions"] = int(user_count)
                state["total_seed_questions"] = int(seed_count)
                state["total_corpus_questions"] = int(corpus_count)
                state["model_version"] = int(state.get("model_version", 0) or 0) + 1
                state["last_trained_at"] = _utc_now_iso()
                state["last_training_seconds"] = training_seconds
                state["is_training"] = False
                state["last_error"] = None
                self._save_state(client_id, bot_id, state)

            final_metrics = {
                "training_seconds": float(training_seconds),
                "model_version": float(state.get("model_version", 0) or 0),
            }
            hist = history.history if hasattr(history, "history") else {}
            epochs_ran = len(hist.get("loss", []) or [])
            if epochs_ran > 0:
                final_metrics["epochs_ran"] = float(epochs_ran)
            for metric_name in ("loss", "val_loss", "perplexity", "val_perplexity"):
                values = hist.get(metric_name) or []
                if values:
                    final_metrics[f"final_{metric_name}"] = float(values[-1])
            mlflow_log_metrics(mlflow_tracker, final_metrics)
            mlflow_set_tags(
                mlflow_tracker,
                {
                    "status": "trained",
                    "model_version": state.get("model_version"),
                },
            )
            mlflow_log_artifact(
                mlflow_tracker,
                bot_model_path,
                artifact_path="autocomplete_model",
            )
            mlflow_log_artifact(
                mlflow_tracker,
                bot_weights_path,
                artifact_path="autocomplete_model",
            )

        # Run DVC tracking after successful bot retraining.
        try:
            dvc_track_autocomplete_training(
                client_id=client_id,
                bot_id=bot_id,
                model_version=int(state.get("model_version", 0) or 0),
            )
        except Exception as err:
            log.warning(
                "DVC autocomplete tracking failed | client_id=%s | bot_id=%s | err=%s",
                client_id,
                bot_id,
                err,
            )

        with self._global_lock:
            self._model_cache.pop(bot_model_path, None)

    def _train_runner(self, client_id: str, bot_id: str) -> None:
        key = self._bot_key(client_id, bot_id)
        try:
            self._train_bot_model(client_id, bot_id)
        except Exception as err:
            lock = self._bot_lock(client_id, bot_id)
            with lock:
                state = self._load_state(client_id, bot_id)
                state["is_training"] = False
                state["last_error"] = str(err)
                self._save_state(client_id, bot_id, state)
        finally:
            with self._global_lock:
                self._queued_airflow.pop(key, None)
                self._inflight.discard(key)

    # -------------------------------
    # Public API expected by app/main.py
    # -------------------------------

    def get_question_rows(
        self,
        client_id: str,
        bot_id: str,
        include_user: bool = True,
        include_seed: bool = True,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        rows = self._load_questions(client_id, bot_id)
        out: list[dict[str, Any]] = []
        for row in rows:
            source = str(row.get("source", "user"))
            if source == "seed" and not include_seed:
                continue
            if source != "seed" and not include_user:
                continue
            out.append(
                {
                    "question": _normalize_text(str(row.get("question", ""))),
                    "ts": float(row.get("ts", 0.0) or 0.0),
                    "source": source,
                }
            )
        if limit and limit > 0 and len(out) > int(limit):
            out = out[-int(limit) :]
        return out

    def build_fallback_seed_questions(self, context_texts: list[str], target_count: int = 50) -> list[str]:
        target_count = max(10, int(target_count))
        terms = _extract_context_terms(context_texts, limit=36)
        rng = random.Random(223)

        templates = [
            "how can i {term}",
            "what is the process for {term}",
            "can you help me with {term}",
            "where can i find details about {term}",
            "is there any update on {term}",
            "please explain {term}",
        ]

        rows: list[str] = []
        for term in terms:
            for template in templates:
                rows.append(_normalize_text(template.format(term=term)))

        generic = _build_seed_queries(size=max(120, target_count * 3), seed=29)
        rows.extend(generic)
        rows = [r for r in rows if r and _count_words(r) >= 3]
        rows = list(dict.fromkeys(rows))
        rng.shuffle(rows)
        return rows[:target_count]

    def bootstrap_seed_questions(
        self,
        client_id: str,
        bot_id: str,
        seed_questions: list[str],
        wait: bool = False,
    ) -> dict[str, Any]:
        normalized = [_normalize_text(q) for q in (seed_questions or [])]
        normalized = [q for q in normalized if q and _count_words(q) >= 3]
        normalized = list(dict.fromkeys(normalized))
        if not normalized:
            return {
                "initialized": False,
                "reason": "no_seed_questions",
                "triggered_training": False,
                "status": self.get_status(client_id, bot_id),
            }

        lock = self._bot_lock(client_id, bot_id)
        with lock:
            state = self._load_state(client_id, bot_id)
            if bool(state.get("bootstrap_completed", False)):
                return {
                    "initialized": False,
                    "reason": "already_bootstrapped",
                    "triggered_training": False,
                    "status": self.get_status(client_id, bot_id),
                }

            existing = self._load_questions(client_id, bot_id)
            # Refresh synthetic seeds while preserving user history.
            user_rows = [row for row in existing if str(row.get("source", "user")) != "seed"]

            now = time.time()
            seed_rows = [
                {
                    "question": q,
                    "ts": now + (idx * 0.001),
                    "source": "seed",
                }
                for idx, q in enumerate(normalized)
            ]

            merged_rows = user_rows + seed_rows
            merged_rows.sort(key=lambda item: float(item.get("ts", now)))
            if len(merged_rows) > self.max_questions_per_bot:
                merged_rows = merged_rows[-self.max_questions_per_bot :]

            self._write_questions(client_id, bot_id, merged_rows)
            try:
                es_replace_bot_questions(client_id, bot_id, merged_rows)
            except Exception:
                pass

            user_count, seed_count = self._count_question_sources(merged_rows)
            state["total_questions"] = int(user_count)
            state["total_seed_questions"] = int(seed_count)
            state["total_corpus_questions"] = int(user_count + seed_count)
            state["bootstrap_completed"] = True
            self._save_state(client_id, bot_id, state)

        triggered = self.trigger_training(
            client_id,
            bot_id,
            wait=wait,
            force=True,
            prefer_airflow=True,
        )
        return {
            "initialized": True,
            "seed_questions_used": len(normalized),
            "triggered_training": bool(triggered),
            "status": self.get_status(client_id, bot_id),
        }

    def get_status(self, client_id: str, bot_id: str) -> dict[str, Any]:
        state = self._load_state(client_id, bot_id)
        questions = self._load_questions(client_id, bot_id)
        user_count, seed_count = self._count_question_sources(questions)

        trained_user = int(
            state.get("last_trained_user_count", state.get("last_trained_count", 0)) or 0
        )

        state["total_questions"] = int(user_count)
        state["total_seed_questions"] = int(seed_count)
        state["total_corpus_questions"] = int(user_count + seed_count)
        state["last_trained_user_count"] = int(trained_user)
        state["last_trained_count"] = int(trained_user)
        state["pending_questions"] = max(0, int(user_count - trained_user))
        state["retrain_every"] = self.retrain_every
        return state

    def record_question(self, client_id: str, bot_id: str, question: str) -> dict[str, Any]:
        normalized = _normalize_text(question)
        if not normalized:
            return {
                "recorded": False,
                "reason": "empty_question",
                "status": self.get_status(client_id, bot_id),
            }

        lock = self._bot_lock(client_id, bot_id)
        deduped = False
        deduped_with = ""
        dedupe_score = 0.0
        with lock:
            state = self._load_state(client_id, bot_id)
            now_ts = float(time.time())
            current_rows = self._load_questions(client_id, bot_id)
            semantic_hit = self._find_semantic_duplicate_user_question(
                normalized,
                current_rows,
            )
            canonical_question = normalized

            if semantic_hit is not None:
                deduped = True
                deduped_with = semantic_hit[0]
                dedupe_score = float(semantic_hit[1])
                canonical_question = deduped_with

                # Keep one canonical row per semantic question group; update recency.
                touched = False
                for idx in range(len(current_rows) - 1, -1, -1):
                    row = current_rows[idx]
                    if str(row.get("source", "user")) == "seed":
                        continue
                    if _normalize_text(str(row.get("question", ""))) != canonical_question:
                        continue
                    current_rows[idx]["ts"] = now_ts
                    touched = True
                    break
                if touched:
                    self._write_questions(client_id, bot_id, current_rows)
                else:
                    self._append_question(
                        client_id,
                        bot_id,
                        canonical_question,
                        source="user",
                        ts=now_ts
                    )
                    current_rows = self._load_questions(client_id, bot_id)
            else:
                self._append_question(
                    client_id,
                    bot_id,
                    normalized,
                    source="user",
                    ts=now_ts
                )
                current_rows = self._load_questions(client_id, bot_id)

            try:
                es_index_question_event(
                    client_id,
                    bot_id,
                    canonical_question,
                    source="user",
                    ts=now_ts,
                    canonical_question=canonical_question,
                )
            except Exception:
                pass

            user_count, seed_count = self._count_question_sources(current_rows)
            state["total_questions"] = int(user_count)
            state["total_seed_questions"] = int(seed_count)
            state["total_corpus_questions"] = int(user_count + seed_count)
            self._save_state(client_id, bot_id, state)

        triggered = False
        if not deduped:
            triggered = self.trigger_training(
                client_id,
                bot_id,
                wait=False,
                force=False,
                prefer_airflow=True,
            )

        return {
            "recorded": True,
            "deduped": deduped,
            "canonical_question": deduped_with or normalized,
            "dedupe_similarity": round(dedupe_score, 4) if deduped else None,
            "retrain_triggered": bool(triggered),
            "status": self.get_status(client_id, bot_id),
        }

    def trigger_training(
        self,
        client_id: str,
        bot_id: str,
        wait: bool = False,
        force: bool = False,
        prefer_airflow: bool = False,
    ) -> bool:
        key = self._bot_key(client_id, bot_id)
        lock = self._bot_lock(client_id, bot_id)
        queue_via_airflow = bool(prefer_airflow and self._airflow_autotrigger and not wait)

        with lock:
            state = self._load_state(client_id, bot_id)
            questions = self._load_questions(client_id, bot_id)
            user_count, seed_count = self._count_question_sources(questions)
            corpus_count = len(questions)

            trained_user = int(
                state.get("last_trained_user_count", state.get("last_trained_count", 0)) or 0
            )
            pending_user = max(0, int(user_count - trained_user))
            model_version = int(state.get("model_version", 0) or 0)
            has_bot_model = os.path.exists(self._model_path(client_id, bot_id))
            has_bot_tokenizer = os.path.exists(self._tokenizer_model_path(client_id, bot_id))

            state["total_questions"] = int(user_count)
            state["total_seed_questions"] = int(seed_count)
            state["total_corpus_questions"] = int(corpus_count)

            if not force:
                if model_version <= 0 or not has_bot_model or not has_bot_tokenizer:
                    if corpus_count < self.min_training_questions:
                        return False
                elif pending_user < self.retrain_every:
                    return False

            try:
                es_replace_bot_questions(client_id, bot_id, questions)
            except Exception:
                pass

            state["is_training"] = True
            state["last_error"] = None
            self._save_state(client_id, bot_id, state)

            if queue_via_airflow:
                now_ts = float(time.time())
                with self._global_lock:
                    self._prune_airflow_queue(now_ts)
                    if key in self._inflight:
                        return False
                    if key in self._queued_airflow:
                        return False
                    self._queued_airflow[key] = now_ts
            else:
                with self._global_lock:
                    self._prune_airflow_queue(float(time.time()))
                    self._queued_airflow.pop(key, None)
                    if key in self._inflight:
                        return False
                    self._inflight.add(key)

        if queue_via_airflow:
            ok, detail = trigger_autocomplete_retrain_dag(
                client_id=client_id,
                bot_id=bot_id,
                reason=("force_training_requested" if force else "pending_question_threshold_met"),
                force=bool(force),
            )
            if ok:
                log.info(
                    "Queued autocomplete retrain via Airflow | client_id=%s | bot_id=%s | detail=%s",
                    client_id,
                    bot_id,
                    detail,
                )
                return True

            with self._global_lock:
                self._queued_airflow.pop(key, None)
            with lock:
                state = self._load_state(client_id, bot_id)
                state["is_training"] = False
                state["last_error"] = f"airflow_trigger_failed: {detail}"[:500]
                self._save_state(client_id, bot_id, state)
            log.warning(
                "Airflow retrain trigger failed | client_id=%s | bot_id=%s | err=%s",
                client_id,
                bot_id,
                detail,
            )
            return False

        worker = threading.Thread(
            target=self._train_runner,
            args=(client_id, bot_id),
            daemon=True,
            name=f"autocomplete-train-{key[0]}-{key[1]}",
        )
        worker.start()

        if wait:
            worker.join()

        return True

    # -------------------------------
    # Inference
    # -------------------------------

    def _load_model_cached(self, path: str) -> keras.Model:
        mtime = float(os.path.getmtime(path))
        with self._global_lock:
            cached = self._model_cache.get(path)
            if cached and cached[0] == mtime:
                return cached[1]

        model = keras.models.load_model(path, compile=False)
        # Warm-up call to avoid first-request latency spike.
        dummy_tokens = np.zeros((1, self.max_seq_len), dtype=np.int32)
        dummy_chars = np.zeros((1, self.max_seq_len, self.max_char_len), dtype=np.int32)
        _ = model([dummy_tokens, dummy_chars], training=False)

        with self._global_lock:
            self._model_cache[path] = (mtime, model)

        return model

    def _active_model(self, client_id: str, bot_id: str) -> keras.Model | None:
        bot_model_path = self._model_path(client_id, bot_id)
        if os.path.exists(bot_model_path):
            return self._load_model_cached(bot_model_path)
        return None

    def _predict_next_probs(
        self,
        model: keras.Model,
        sp: spm.SentencePieceProcessor,
        token_ids: list[int],
    ) -> np.ndarray:
        batch = self._predict_batch_probs(model, sp, [token_ids])
        return batch[0]

    def _predict_batch_probs(
        self,
        model: keras.Model,
        sp: spm.SentencePieceProcessor,
        token_sequences: list[list[int]],
    ) -> np.ndarray:
        batch_size = len(token_sequences)
        if batch_size == 0:
            return np.zeros((0, 0), dtype=np.float64)

        token_arr = np.zeros((batch_size, self.max_seq_len), dtype=np.int32)
        char_arr = np.zeros((batch_size, self.max_seq_len, self.max_char_len), dtype=np.int32)

        for row_idx, token_ids in enumerate(token_sequences):
            prefix = token_ids[-self.max_seq_len :]
            if prefix:
                token_arr[row_idx, -len(prefix) :] = np.array(prefix, dtype=np.int32)

            start = self.max_seq_len - len(prefix)
            for j, token_id in enumerate(prefix):
                piece = self._piece_text(sp, int(token_id))
                char_arr[row_idx, start + j] = np.array(
                    self._encode_char_token(piece), dtype=np.int32
                )

        # Direct call avoids per-request predict() overhead.
        logits = model([token_arr, char_arr], training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy().astype(np.float64)
        return probs

    def _beam_suggestions(
        self,
        model: keras.Model,
        sp: spm.SentencePieceProcessor,
        text: str,
        max_suggestions: int,
        max_future_words: int,
    ) -> list[dict[str, Any]]:
        text = text or ""
        input_ids = sp.encode(text, out_type=int)

        beam_width = min(6, max(3, max_suggestions + 1))
        step_k = min(5, max(3, max_suggestions + 1))

        beams: list[tuple[list[int], float, bool]] = [([], 0.0, False)]

        for step in range(max(1, int(max_future_words))):
            next_beams: list[tuple[list[int], float, bool]] = []
            active: list[tuple[list[int], float]] = []

            for gen_ids, logp, done in beams:
                if done:
                    next_beams.append((gen_ids, logp, True))
                    continue
                active.append((gen_ids, logp))

            probs_batch = self._predict_batch_probs(
                model,
                sp,
                [input_ids + gen_ids for gen_ids, _ in active],
            )

            for (gen_ids, logp), probs in zip(active, probs_batch):
                top_ids = probs.argsort()[-step_k:][::-1]
                best_prob = float(probs[top_ids[0]]) if len(top_ids) else 0.0

                if gen_ids and best_prob < 0.18:
                    next_beams.append((gen_ids, logp, True))
                    continue

                expanded = False
                for token_id in top_ids:
                    prob = float(probs[int(token_id)])
                    if prob < 0.03:
                        continue
                    if not self._is_valid_generation_piece(sp, int(token_id)):
                        continue

                    expanded = True
                    next_beams.append(
                        (
                            gen_ids + [int(token_id)],
                            logp + math.log(max(prob, 1e-9)),
                            False,
                        )
                    )

                if gen_ids:
                    # stop option for variable-length outputs
                    next_beams.append((gen_ids, logp - 0.05, True))

                if not expanded and not gen_ids:
                    next_beams.append((gen_ids, logp, True))

            next_beams.sort(
                key=lambda item: (item[1] / max(1, len(item[0])), item[1]),
                reverse=True,
            )
            beams = next_beams[:beam_width]

            if beams and all(done for _, _, done in beams):
                break

        raw_input = _normalize_text(text)
        seen: set[str] = set()
        results: list[dict[str, Any]] = []

        for gen_ids, logp, _ in beams:
            if not gen_ids:
                continue

            full_text = _normalize_text(sp.decode(input_ids + gen_ids))
            if not full_text:
                continue
            if full_text == raw_input:
                continue
            if full_text in seen:
                continue

            suffix_text = _normalize_text(sp.decode(gen_ids))
            future_words = max(1, _count_words(suffix_text))

            confidence = float(math.exp(logp / max(1, len(gen_ids))))
            if confidence < 0.12:
                continue

            seen.add(full_text)
            results.append(
                {
                    "text": full_text,
                    "confidence": round(confidence, 4),
                    "future_words": int(future_words),
                }
            )

        results.sort(key=lambda item: item["confidence"], reverse=True)

        if results:
            return results[: max(1, int(max_suggestions))]

        # fallback: one-step top-k suggestions
        probs = self._predict_next_probs(model, sp, input_ids)
        top_ids = probs.argsort()[-max(1, int(max_suggestions)) :][::-1]

        fallback: list[dict[str, Any]] = []
        for token_id in top_ids:
            token_id = int(token_id)
            if not self._is_valid_generation_piece(sp, token_id):
                continue

            full_text = _normalize_text(sp.decode(input_ids + [token_id]))
            if not full_text or full_text == raw_input:
                continue

            fallback.append(
                {
                    "text": full_text,
                    "confidence": round(float(probs[token_id]), 4),
                    "future_words": 1,
                }
            )

        fallback = fallback[: max(1, int(max_suggestions))]
        if fallback:
            return fallback

        # Last-resort deterministic fallback to keep UI stable.
        heuristic: list[dict[str, Any]] = []
        if raw_input:
            for obj in BASE_OBJECTS:
                text_candidate = _normalize_text(f"{raw_input} {obj}")
                if text_candidate and text_candidate != raw_input:
                    heuristic.append(
                        {
                            "text": text_candidate,
                            "confidence": 0.08,
                            "future_words": max(1, _count_words(obj)),
                        }
                    )
                if len(heuristic) >= max(1, int(max_suggestions)):
                    break

        return heuristic

    def _suggest_cache_get(self, key: str) -> list[dict[str, Any]] | None:
        with self._global_lock:
            entry = self._suggest_cache.get(key)
            if not entry:
                return None
            ts, items = entry
            if (time.time() - ts) * 1000.0 > self._suggest_cache_ttl_ms:
                self._suggest_cache.pop(key, None)
                return None
            return [dict(item) for item in items]

    def _suggest_cache_set(self, key: str, items: list[dict[str, Any]]) -> None:
        with self._global_lock:
            self._suggest_cache[key] = (time.time(), [dict(item) for item in items])
            if len(self._suggest_cache) > self._suggest_cache_max:
                oldest_key = next(iter(self._suggest_cache.keys()), None)
                if oldest_key is not None:
                    self._suggest_cache.pop(oldest_key, None)

    def _clip_suggestion_to_future_words(
        self,
        query_text: str,
        candidate_text: str,
        max_future_words: int,
    ) -> tuple[str, int] | None:
        query = _normalize_text(query_text)
        candidate = _normalize_text(candidate_text)
        if not candidate:
            return None

        q_words = query.split()
        c_words = candidate.split()
        if not c_words:
            return None

        if q_words and c_words[: len(q_words)] == q_words:
            tail = c_words[len(q_words) :]
            if not tail:
                return None
            tail = tail[: max(1, int(max_future_words))]
            text = " ".join(q_words + tail).strip()
            return text, int(max(1, len(tail)))

        return None

    def _merge_hybrid_suggestions(
        self,
        query_text: str,
        model_suggestions: list[dict[str, Any]],
        es_suggestions: list[dict[str, Any]],
        *,
        max_suggestions: int,
        max_future_words: int,
    ) -> list[dict[str, Any]]:
        normalized_query = _normalize_text(query_text)
        by_text: dict[str, dict[str, Any]] = {}

        for item in model_suggestions or []:
            text_raw = _normalize_text(str(item.get("text", "")))
            if not text_raw or text_raw == normalized_query:
                continue
            clipped = self._clip_suggestion_to_future_words(
                normalized_query,
                text_raw,
                max_future_words=max_future_words,
            )
            if not clipped:
                continue
            text, future_words = clipped
            if text == normalized_query:
                continue
            row = by_text.setdefault(
                text,
                {
                    "text": text,
                    "model_score": 0.0,
                    "es_score": 0.0,
                    "future_words": int(future_words),
                },
            )
            row["model_score"] = max(float(row["model_score"]), float(item.get("confidence", 0.0) or 0.0))
            row["future_words"] = int(max(int(row["future_words"]), int(future_words)))

        es_max = 0.0
        for item in es_suggestions or []:
            es_max = max(es_max, float(item.get("score", 0.0) or 0.0))
        es_max = max(es_max, 1e-9)

        for item in es_suggestions or []:
            text_raw = _normalize_text(str(item.get("text", "")))
            if not text_raw or text_raw == normalized_query:
                continue
            clipped = self._clip_suggestion_to_future_words(
                normalized_query,
                text_raw,
                max_future_words=max_future_words,
            )
            if not clipped:
                continue
            text, future_words = clipped
            if text == normalized_query:
                continue
            norm_score = max(0.0, min(1.0, float(item.get("score", 0.0) or 0.0) / es_max))
            row = by_text.setdefault(
                text,
                {
                    "text": text,
                    "model_score": 0.0,
                    "es_score": 0.0,
                    "future_words": int(future_words),
                },
            )
            row["es_score"] = max(float(row["es_score"]), norm_score)
            row["future_words"] = int(max(int(row["future_words"]), int(future_words)))

        merged: list[dict[str, Any]] = []
        for row in by_text.values():
            model_score = float(row.get("model_score", 0.0) or 0.0)
            es_score = float(row.get("es_score", 0.0) or 0.0)

            if model_score > 0.0 and es_score > 0.0:
                confidence = (AUTOCOMPLETE_MODEL_WEIGHT * model_score) + (AUTOCOMPLETE_ES_WEIGHT * es_score)
            elif model_score > 0.0:
                confidence = model_score
            else:
                confidence = AUTOCOMPLETE_ES_WEIGHT * es_score

            merged.append(
                {
                    "text": str(row["text"]),
                    "confidence": round(float(confidence), 4),
                    "future_words": int(max(1, min(int(row.get("future_words", 1)), int(max_future_words)))),
                }
            )

        merged.sort(key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)
        return merged[: max(1, int(max_suggestions))]

    def _independent_dual_suggestions(
        self,
        query_text: str,
        model_suggestions: list[dict[str, Any]],
        es_suggestions: list[dict[str, Any]],
        *,
        max_future_words: int,
    ) -> list[dict[str, Any]]:
        normalized_query = _normalize_text(query_text)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        # 1) One model suggestion
        for item in model_suggestions or []:
            text_raw = _normalize_text(str(item.get("text", "")))
            if not text_raw or text_raw == normalized_query:
                continue
            clipped = self._clip_suggestion_to_future_words(
                normalized_query,
                text_raw,
                max_future_words=max_future_words,
            )
            if not clipped:
                continue
            text_candidate, fw = clipped
            if text_candidate in seen:
                continue
            seen.add(text_candidate)
            out.append(
                {
                    "text": text_candidate,
                    "confidence": round(float(item.get("confidence", 0.0) or 0.0), 4),
                    "future_words": int(max(1, fw)),
                    "strategy": "model",
                }
            )
            break

        # 2) One BM25/ES suggestion
        for item in es_suggestions or []:
            text_raw = _normalize_text(str(item.get("text", "")))
            if not text_raw or text_raw == normalized_query:
                continue
            clipped = self._clip_suggestion_to_future_words(
                normalized_query,
                text_raw,
                max_future_words=max_future_words,
            )
            if not clipped:
                continue
            text_candidate, fw = clipped
            if text_candidate in seen:
                continue
            seen.add(text_candidate)
            out.append(
                {
                    "text": text_candidate,
                    # Keep ES score display stable; full ranking already done in ES.
                    "confidence": round(float(item.get("score", 0.0) or 0.0), 4),
                    "future_words": int(max(1, fw)),
                    "strategy": "bm25",
                }
            )
            break

        return out[:2]

    def suggest(
        self,
        client_id: str,
        bot_id: str,
        text: str,
        max_suggestions: int = 5,
        max_future_words: int = 5,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        normalized_query = _normalize_text(text)
        max_suggestions = max(1, min(int(max_suggestions), 3))
        max_future_words = max(1, min(int(max_future_words), 3))

        status = self.get_status(client_id, bot_id)
        model_version = int(status.get("model_version", 0) or 0)

        if len(normalized_query) < 4:
            result = {
                "suggestions": [],
                "status": {
                    "model_version": model_version,
                    "is_training": bool(status.get("is_training", False)),
                    "pending_questions": status.get("pending_questions", 0),
                    "last_trained_at": status.get("last_trained_at"),
                },
            }
            record_autocomplete_query_event(
                client_id=client_id,
                bot_id=bot_id,
                suggestions=result.get("suggestions", []),
                reason="query_too_short",
            )
            return result

        dual_mode = "dualv1"
        cache_key = "|".join(
            [
                _safe_id(client_id),
                _safe_id(bot_id),
                str(model_version),
                dual_mode,
                normalized_query,
                str(max_suggestions),
                str(max_future_words),
            ]
        )
        cached = self._suggest_cache_get(cache_key)
        if cached is not None:
            log.debug(
                "Autocomplete suggest cache hit | client_id=%s | bot_id=%s | query='%s' | results=%d",
                client_id,
                bot_id,
                normalized_query[:72],
                len(cached),
            )
            result = {
                "suggestions": cached,
                "status": {
                    "model_version": model_version,
                    "is_training": bool(status.get("is_training", False)),
                    "pending_questions": status.get("pending_questions", 0),
                    "last_trained_at": status.get("last_trained_at"),
                },
            }
            record_autocomplete_query_event(
                client_id=client_id,
                bot_id=bot_id,
                suggestions=result.get("suggestions", []),
                reason="cache_hit",
            )
            return result

        sp = self._load_bot_sp(client_id, bot_id)
        model = self._active_model(client_id, bot_id)

        model_suggestions: list[dict[str, Any]] = []
        if sp is not None and model is not None:
            model_suggestions = self._beam_suggestions(
                model=model,
                sp=sp,
                text=normalized_query,
                max_suggestions=1,
                max_future_words=max_future_words,
            )

        es_suggestions: list[dict[str, Any]] = []
        try:
            es_suggestions = es_search_question_suggestions(
                client_id=client_id,
                bot_id=bot_id,
                query_text=normalized_query,
                top_k=4,
            )
        except Exception:
            es_suggestions = []

        suggestions = self._independent_dual_suggestions(
            query_text=normalized_query,
            model_suggestions=model_suggestions,
            es_suggestions=es_suggestions,
            max_future_words=max_future_words,
        )
        if not suggestions:
            if model_suggestions:
                suggestions = model_suggestions[:1]
            elif es_suggestions:
                suggestions = [
                    {
                        "text": str(es_suggestions[0].get("text", "")),
                        "confidence": round(float(es_suggestions[0].get("score", 0.0) or 0.0), 4),
                        "future_words": max(1, _count_words(str(es_suggestions[0].get("text", "")))),
                        "strategy": "bm25",
                    }
                ]

        log.debug(
            (
                "Autocomplete hybrid summary | client_id=%s | bot_id=%s | query='%s' | "
                "model_candidates=%d | es_candidates=%d | final=%d | mode=independent_dual | model_ready=%s | tokenizer_ready=%s | took_ms=%.2f"
            ),
            client_id,
            bot_id,
            normalized_query[:72],
            len(model_suggestions),
            len(es_suggestions),
            len(suggestions),
            bool(model is not None),
            bool(sp is not None),
            (time.perf_counter() - started) * 1000.0,
        )

        self._suggest_cache_set(cache_key, suggestions)

        result = {
            "suggestions": suggestions,
            "status": {
                "model_version": model_version,
                "is_training": bool(status.get("is_training", False)),
                "pending_questions": status.get("pending_questions", 0),
                "last_trained_at": status.get("last_trained_at"),
            },
        }
        record_autocomplete_query_event(
            client_id=client_id,
            bot_id=bot_id,
            suggestions=result.get("suggestions", []),
            reason="fresh_inference",
        )
        return result


_MANAGER: BotAutocompleteManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_autocomplete_manager() -> BotAutocompleteManager:
    global _MANAGER
    if _MANAGER is not None:
        return _MANAGER

    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = BotAutocompleteManager()

    return _MANAGER
