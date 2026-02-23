from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterable

from app.pipeline.logger import get_logger

try:
    from elasticsearch import Elasticsearch, helpers
except Exception:
    Elasticsearch = None
    helpers = None


log = get_logger("elasticsearch")

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_id(raw: str) -> str:
    return SAFE_ID_RE.sub("_", str(raw or "")).strip("_") or "unknown"


def _normalize_text(text: str) -> str:
    return " ".join(WORD_RE.findall((text or "").lower()))


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _expand_bm25_query(query_text: str, source_filter: Iterable[str] | str | None = None) -> str:
    q = _clean_text(query_text).lower()
    if not q:
        return ""

    extras: list[str] = []
    if "about" in q or "company" in q or "overview" in q:
        extras.extend(["about us", "who we are", "our company", "our story"])

    if any(term in q for term in ("contact", "phone", "email", "address", "reach", "call")):
        extras.extend(["contact us", "phone number", "email address", "office address", "support"])

    if any(term in q for term in ("service", "services", "offer", "offering")):
        extras.extend(["what we do", "our services", "solutions"])

    if source_filter is not None:
        if isinstance(source_filter, (list, tuple, set)):
            sf = {str(v).strip().lower() for v in source_filter if str(v).strip()}
        else:
            sf = {str(source_filter).strip().lower()}
        if "website" in sf:
            extras.extend(["website", "page", "site"])

    if extras:
        return " ".join(dict.fromkeys([q] + extras))
    return q


def _normalize_weights(a: float, b: float, default_a: float, default_b: float) -> tuple[float, float]:
    a = max(0.0, float(a))
    b = max(0.0, float(b))
    total = a + b
    if total <= 0:
        return float(default_a), float(default_b)
    return float(a / total), float(b / total)


ES_ENABLED = _env_bool("ES_ENABLED", True)
ES_URLS = [
    url.strip()
    for url in str(os.getenv("ES_URLS", "http://localhost:9200")).split(",")
    if url.strip()
]
ES_USERNAME = str(os.getenv("ES_USERNAME", "")).strip()
ES_PASSWORD = str(os.getenv("ES_PASSWORD", "")).strip()
ES_API_KEY = str(os.getenv("ES_API_KEY", "")).strip()
ES_VERIFY_CERTS = _env_bool("ES_VERIFY_CERTS", False)
ES_CA_CERTS = str(os.getenv("ES_CA_CERTS", "")).strip() or None

ES_REQUEST_TIMEOUT = max(1.0, _env_float("ES_REQUEST_TIMEOUT_SEC", 10.0))
ES_PING_TIMEOUT = max(0.5, _env_float("ES_PING_TIMEOUT_SEC", 3.0))
ES_COOLDOWN_SECONDS = max(5.0, _env_float("ES_COOLDOWN_SECONDS", 20.0))

ES_CHUNK_INDEX = str(os.getenv("ES_CHUNK_INDEX", "bot_chunks_hybrid_v1")).strip()
ES_QUESTION_INDEX = str(os.getenv("ES_QUESTION_INDEX", "bot_questions_hybrid_v1")).strip()

_sem_w_raw = _env_float("HYBRID_SEMANTIC_WEIGHT", 0.45)
_bm25_w_raw = _env_float("HYBRID_BM25_WEIGHT", 0.55)
HYBRID_SEMANTIC_WEIGHT, HYBRID_BM25_WEIGHT = _normalize_weights(
    _sem_w_raw,
    _bm25_w_raw,
    default_a=0.45,
    default_b=0.55,
)

_ac_model_w_raw = _env_float("AUTOCOMPLETE_MODEL_WEIGHT", 0.35)
_ac_es_w_raw = _env_float("AUTOCOMPLETE_ES_WEIGHT", 0.65)
AUTOCOMPLETE_MODEL_WEIGHT, AUTOCOMPLETE_ES_WEIGHT = _normalize_weights(
    _ac_model_w_raw,
    _ac_es_w_raw,
    default_a=0.35,
    default_b=0.65,
)

log.info(
    "ES hybrid config | enabled=%s | urls=%s | retrieval_weights=semantic:%.2f bm25:%.2f | autocomplete_weights=model:%.2f es:%.2f",
    bool(ES_ENABLED),
    ",".join(ES_URLS),
    HYBRID_SEMANTIC_WEIGHT,
    HYBRID_BM25_WEIGHT,
    AUTOCOMPLETE_MODEL_WEIGHT,
    AUTOCOMPLETE_ES_WEIGHT,
)


_CLIENT: Any = None
_CLIENT_LOCK = threading.Lock()
_DISABLED_UNTIL_TS = 0.0
_LAST_FAILURE_LOG_TS = 0.0
_INDEX_READY: set[str] = set()


def _mark_failure(action: str, err: Exception) -> None:
    global _CLIENT, _DISABLED_UNTIL_TS, _LAST_FAILURE_LOG_TS
    _CLIENT = None
    _DISABLED_UNTIL_TS = time.time() + ES_COOLDOWN_SECONDS
    now = time.time()
    if now - _LAST_FAILURE_LOG_TS >= 5.0:
        log.warning("Elasticsearch %s failed: %s", action, err)
        _LAST_FAILURE_LOG_TS = now


def _ping(client: Any) -> bool:
    try:
        return bool(client.ping(request_timeout=ES_PING_TIMEOUT))
    except TypeError:
        return bool(client.ping())


def _build_client() -> Any | None:
    if Elasticsearch is None or not ES_ENABLED:
        return None

    kwargs: dict[str, Any] = {
        "hosts": ES_URLS,
        "request_timeout": ES_REQUEST_TIMEOUT,
        "max_retries": 1,
        "retry_on_timeout": False,
        "verify_certs": ES_VERIFY_CERTS,
    }
    if ES_CA_CERTS:
        kwargs["ca_certs"] = ES_CA_CERTS
    if ES_API_KEY:
        kwargs["api_key"] = ES_API_KEY
    elif ES_USERNAME and ES_PASSWORD:
        kwargs["basic_auth"] = (ES_USERNAME, ES_PASSWORD)

    try:
        client = Elasticsearch(**kwargs)
        if not _ping(client):
            raise RuntimeError("ping_failed")
        return client
    except Exception as err:
        _mark_failure("connect", err)
        return None


def is_enabled() -> bool:
    return bool(ES_ENABLED and Elasticsearch is not None)


def get_client() -> Any | None:
    global _CLIENT
    if not ES_ENABLED or Elasticsearch is None:
        return None
    if time.time() < _DISABLED_UNTIL_TS:
        return None

    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        _CLIENT = _build_client()
        return _CLIENT


def _index_exists(client: Any, index_name: str) -> bool:
    try:
        exists = client.indices.exists(index=index_name)
        return bool(exists)
    except Exception:
        return False


def _ensure_chunk_index(client: Any) -> bool:
    if not client:
        return False
    if ES_CHUNK_INDEX in _INDEX_READY:
        return True

    body = {
        # Keep index creation serverless-compatible: serverless Elasticsearch
        # rejects explicit shard/replica settings.
        "mappings": {
            "properties": {
                "client_id": {"type": "keyword"},
                "bot_id": {"type": "keyword"},
                "cluster": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "chunk_ref": {"type": "keyword"},
                "text": {"type": "text"},
                "topic": {"type": "text"},
                "source_type": {"type": "keyword"},
                "source_url": {"type": "keyword"},
                "pdf": {"type": "keyword"},
                "updated_at": {"type": "date"},
            }
        },
    }

    try:
        if not _index_exists(client, ES_CHUNK_INDEX):
            client.indices.create(index=ES_CHUNK_INDEX, body=body)
        _INDEX_READY.add(ES_CHUNK_INDEX)
        return True
    except Exception as err:
        _mark_failure("ensure_chunk_index", err)
        return False


def _ensure_question_index(client: Any) -> bool:
    if not client:
        return False
    if ES_QUESTION_INDEX in _INDEX_READY:
        return True

    body = {
        # Keep index creation serverless-compatible: serverless Elasticsearch
        # rejects explicit shard/replica settings.
        "mappings": {
            "properties": {
                "client_id": {"type": "keyword"},
                "bot_id": {"type": "keyword"},
                "text": {"type": "text"},
                "canonical_question": {"type": "keyword"},
                "suggest": {"type": "completion"},
                "source": {"type": "keyword"},
                "ask_count": {"type": "integer"},
                "ts": {"type": "date"},
                "updated_at": {"type": "date"},
            }
        },
    }

    try:
        if not _index_exists(client, ES_QUESTION_INDEX):
            client.indices.create(index=ES_QUESTION_INDEX, body=body)
        else:
            # Ensure completion field exists for prefix suggestion strategy.
            client.indices.put_mapping(
                index=ES_QUESTION_INDEX,
                properties={
                    "suggest": {"type": "completion"},
                    "canonical_question": {"type": "keyword"},
                    "ask_count": {"type": "integer"},
                },
            )
        _INDEX_READY.add(ES_QUESTION_INDEX)
        return True
    except Exception as err:
        _mark_failure("ensure_question_index", err)
        return False


def _delete_bot_docs(client: Any, index_name: str, client_id: str, bot_id: str) -> bool:
    if not client:
        return False
    if not _index_exists(client, index_name):
        return True
    try:
        client.delete_by_query(
            index=index_name,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"client_id": str(client_id)}},
                            {"term": {"bot_id": str(bot_id)}},
                        ]
                    }
                }
            },
            conflicts="proceed",
            refresh=False,
            ignore_unavailable=True,
            request_timeout=max(1.0, ES_REQUEST_TIMEOUT),
        )
        return True
    except Exception as err:
        _mark_failure("delete_by_query", err)
        return False


def _iso_from_ts(ts: float) -> str:
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception:
        dt = datetime.now(timezone.utc)
    return dt.isoformat()


def _question_doc_id(
    client_id: str,
    bot_id: str,
    question: str,
    source: str,
) -> str:
    seed = "|".join(
        [
            _safe_id(client_id),
            _safe_id(bot_id),
            _normalize_text(question),
            str(source),
        ]
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def index_bot_chunks(client_id: str, bot_id: str, chunks: list[dict[str, Any]]) -> bool:
    client = get_client()
    if not client:
        return False
    if not _ensure_chunk_index(client):
        return False
    if not _delete_bot_docs(client, ES_CHUNK_INDEX, client_id, bot_id):
        return False
    if not chunks:
        return True
    if helpers is None:
        return False

    actions = _build_chunk_actions(client_id, bot_id, chunks)

    if not actions:
        return True

    try:
        helpers.bulk(
            client,
            actions,
            raise_on_error=False,
            request_timeout=max(1.0, ES_REQUEST_TIMEOUT),
        )
        log.info(
            "ES chunk sync complete | client_id=%s | bot_id=%s | indexed=%d | index=%s",
            client_id,
            bot_id,
            len(actions),
            ES_CHUNK_INDEX,
        )
        return True
    except Exception as err:
        _mark_failure("bulk_index_chunks", err)
        return False


def _build_chunk_actions(client_id: str, bot_id: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for row in chunks:
        text = _clean_text(row.get("text", ""))
        if not text:
            continue
        cluster = str(row.get("cluster", ""))
        chunk_index = int(row.get("chunk_index", 0) or 0)
        chunk_ref = str(row.get("chunk_ref") or f"{cluster}_{chunk_index}")
        doc_id = "|".join(
            [
                _safe_id(client_id),
                _safe_id(bot_id),
                _safe_id(chunk_ref),
            ]
        )
        actions.append(
            {
                "_op_type": "index",
                "_index": ES_CHUNK_INDEX,
                "_id": doc_id,
                "_source": {
                    "client_id": str(client_id),
                    "bot_id": str(bot_id),
                    "cluster": cluster,
                    "chunk_index": chunk_index,
                    "chunk_ref": chunk_ref,
                    "text": text,
                    "topic": _clean_text(row.get("topic", "")),
                    "source_type": str(row.get("source_type", "")),
                    "source_url": str(row.get("source_url", "")),
                    "pdf": str(row.get("pdf", "")),
                    "updated_at": now_iso,
                },
            }
        )
    return actions


def append_bot_chunks(client_id: str, bot_id: str, chunks: list[dict[str, Any]]) -> bool:
    client = get_client()
    if not client:
        return False
    if not _ensure_chunk_index(client):
        return False
    if helpers is None:
        return False

    actions = _build_chunk_actions(client_id, bot_id, chunks)
    if not actions:
        return True

    try:
        helpers.bulk(
            client,
            actions,
            raise_on_error=False,
            request_timeout=max(1.0, ES_REQUEST_TIMEOUT),
        )
        log.info(
            "ES chunk append complete | client_id=%s | bot_id=%s | indexed=%d | index=%s",
            client_id,
            bot_id,
            len(actions),
            ES_CHUNK_INDEX,
        )
        return True
    except Exception as err:
        _mark_failure("bulk_append_chunks", err)
        return False


def delete_bot_chunks(client_id: str, bot_id: str) -> bool:
    client = get_client()
    if not client:
        return False
    return _delete_bot_docs(client, ES_CHUNK_INDEX, client_id, bot_id)


def search_bm25_chunks(
    query_text: str,
    client_id: str,
    bot_id: str,
    *,
    top_k: int = 8,
    source_filter: Iterable[str] | str | None = None,
    cluster_filter: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    query_text = _clean_text(query_text)
    if not query_text:
        return []
    expanded_query = _expand_bm25_query(query_text, source_filter=source_filter)

    client = get_client()
    if not client:
        return []
    if not _ensure_chunk_index(client):
        return []

    filters: list[dict[str, Any]] = [
        {"term": {"client_id": str(client_id)}},
        {"term": {"bot_id": str(bot_id)}},
    ]

    if source_filter:
        if isinstance(source_filter, (list, tuple, set)):
            allowed = [str(x) for x in source_filter if str(x).strip()]
            if allowed:
                filters.append({"terms": {"source_type": allowed}})
        else:
            filters.append({"term": {"source_type": str(source_filter)}})

    if cluster_filter:
        cluster_vals = [str(x) for x in cluster_filter if str(x).strip()]
        if cluster_vals:
            filters.append({"terms": {"cluster": cluster_vals}})

    body = {
        "size": max(1, int(top_k)),
        "_source": [
            "text",
            "topic",
            "cluster",
            "chunk_index",
            "chunk_ref",
            "source_type",
            "source_url",
            "pdf",
        ],
        "query": {
            "bool": {
                "filter": filters,
                "should": [
                    {"match_phrase": {"text": {"query": query_text, "boost": 6.5}}},
                    {"match_phrase_prefix": {"text": {"query": query_text, "boost": 5.0}}},
                    {
                        "multi_match": {
                            "query": expanded_query,
                            "fields": ["text^6", "topic^2.5", "pdf^1.2", "source_url^1.2"],
                            "type": "best_fields",
                            "operator": "and",
                            "boost": 4.0,
                        }
                    },
                    {
                        "multi_match": {
                            "query": expanded_query,
                            "fields": ["text^4", "topic^2", "pdf", "source_url"],
                            "type": "best_fields",
                            "operator": "or",
                            "boost": 2.5,
                        }
                    },
                    {"match_bool_prefix": {"text": {"query": query_text, "boost": 2.2}}},
                    {
                        "match": {
                            "text": {
                                "query": query_text,
                                "fuzziness": "AUTO",
                                "prefix_length": 1,
                                "boost": 1.4,
                            }
                        }
                    },
                    {
                        "simple_query_string": {
                            "query": expanded_query,
                            "fields": ["text^3", "topic^1.8", "source_url"],
                            "default_operator": "or",
                            "boost": 1.2,
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
    }

    try:
        response = client.search(
            index=ES_CHUNK_INDEX,
            body=body,
            request_timeout=ES_REQUEST_TIMEOUT,
        )
        hits = (((response or {}).get("hits") or {}).get("hits") or [])
    except Exception as err:
        _mark_failure("search_bm25_chunks", err)
        return []

    if not hits:
        # Lower-precision fallback for sparse/short website content.
        fallback_body = {
            "size": max(1, int(top_k)),
            "_source": body["_source"],
            "query": {
                "bool": {
                    "filter": filters,
                    "must": [
                        {
                            "simple_query_string": {
                                "query": expanded_query,
                                "fields": ["text^3", "topic^1.5", "pdf", "source_url"],
                                "default_operator": "or",
                            }
                        }
                    ],
                }
            },
        }
        try:
            response = client.search(
                index=ES_CHUNK_INDEX,
                body=fallback_body,
                request_timeout=ES_REQUEST_TIMEOUT,
            )
            hits = (((response or {}).get("hits") or {}).get("hits") or [])
        except Exception:
            hits = []

    out: list[dict[str, Any]] = []
    for hit in hits:
        source = hit.get("_source") or {}
        score = float(hit.get("_score", 0.0) or 0.0)
        text = _clean_text(source.get("text", ""))
        if not text:
            continue
        cluster = source.get("cluster")
        chunk_index = int(source.get("chunk_index", 0) or 0)
        chunk_ref = str(source.get("chunk_ref") or f"{cluster}_{chunk_index}")
        out.append(
            {
                "text": text,
                "topic": source.get("topic"),
                "cluster": cluster,
                "chunk_index": chunk_index,
                "chunk_ref": chunk_ref,
                "score": score,
                "source_type": source.get("source_type"),
                "source_url": source.get("source_url"),
                "pdf": source.get("pdf"),
            }
        )
    log.debug(
        "ES BM25 chunk search | client_id=%s | bot_id=%s | query='%s' | hits=%d | top_k=%d | took_ms=%.2f",
        client_id,
        bot_id,
        query_text[:72],
        len(out),
        int(top_k),
        (time.perf_counter() - start) * 1000.0,
    )
    return out


def replace_bot_questions(client_id: str, bot_id: str, rows: list[dict[str, Any]]) -> bool:
    client = get_client()
    if not client:
        return False
    if not _ensure_question_index(client):
        return False
    if not _delete_bot_docs(client, ES_QUESTION_INDEX, client_id, bot_id):
        return False
    if not rows:
        return True
    if helpers is None:
        return False

    actions = []
    now_iso = datetime.now(timezone.utc).isoformat()
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        text = _normalize_text(str(row.get("question", row.get("text", ""))))
        if not text:
            continue
        source = "seed" if str(row.get("source", "user")) == "seed" else "user"
        ts = float(row.get("ts", time.time()) or time.time())
        key = (text, source)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = {"ts": ts, "count": 1}
        else:
            existing["count"] = int(existing.get("count", 1)) + 1
            if ts > float(existing.get("ts", 0.0) or 0.0):
                existing["ts"] = ts

    for (text, source), payload in deduped.items():
        ask_count = max(1, int(payload.get("count", 1)))
        ts = float(payload.get("ts", time.time()) or time.time())
        actions.append(
            {
                "_op_type": "index",
                "_index": ES_QUESTION_INDEX,
                "_id": _question_doc_id(client_id, bot_id, text, source),
                "_source": {
                    "client_id": str(client_id),
                    "bot_id": str(bot_id),
                    "text": text,
                    "canonical_question": text,
                    "suggest": {
                        "input": [text],
                        "weight": (8 if source == "user" else 4) + min(24, ask_count),
                    },
                    "source": source,
                    "ask_count": ask_count,
                    "ts": _iso_from_ts(ts),
                    "updated_at": now_iso,
                },
            }
        )

    if not actions:
        return True

    try:
        helpers.bulk(
            client,
            actions,
            raise_on_error=False,
            request_timeout=max(1.0, ES_REQUEST_TIMEOUT),
        )
        log.info(
            "ES question corpus sync | client_id=%s | bot_id=%s | indexed=%d | index=%s",
            client_id,
            bot_id,
            len(actions),
            ES_QUESTION_INDEX,
        )
        return True
    except Exception as err:
        _mark_failure("bulk_index_questions", err)
        return False


def delete_bot_questions(client_id: str, bot_id: str) -> bool:
    client = get_client()
    if not client:
        return False
    return _delete_bot_docs(client, ES_QUESTION_INDEX, client_id, bot_id)


def index_question_event(
    client_id: str,
    bot_id: str,
    question: str,
    *,
    source: str = "user",
    ts: float | None = None,
    canonical_question: str | None = None,
) -> bool:
    text = _normalize_text(canonical_question or question)
    if not text:
        return False

    client = get_client()
    if not client:
        return False
    if not _ensure_question_index(client):
        return False

    now_ts = float(ts if ts is not None else time.time())
    now_iso = datetime.now(timezone.utc).isoformat()
    source_val = "seed" if str(source) == "seed" else "user"
    doc_id = _question_doc_id(
        client_id,
        bot_id,
        text,
        source_val,
    )

    payload = {
        "client_id": str(client_id),
        "bot_id": str(bot_id),
        "text": text,
        "canonical_question": text,
        "suggest": {
            "input": [text],
            "weight": 9 if source_val == "user" else 5,
        },
        "source": source_val,
        "ask_count": 1,
        "ts": _iso_from_ts(now_ts),
        "updated_at": now_iso,
    }

    try:
        client.update(
            index=ES_QUESTION_INDEX,
            id=doc_id,
            body={
                "script": {
                    "lang": "painless",
                    "source": (
                        "ctx._source.client_id = params.client_id; "
                        "ctx._source.bot_id = params.bot_id; "
                        "ctx._source.text = params.text; "
                        "ctx._source.canonical_question = params.canonical_question; "
                        "ctx._source.source = params.source; "
                        "ctx._source.ts = params.ts; "
                        "ctx._source.updated_at = params.updated_at; "
                        "if (ctx._source.ask_count == null) { ctx._source.ask_count = 0; } "
                        "ctx._source.ask_count += 1; "
                        "int base = params.base_weight; "
                        "int capped = Math.min(24, ctx._source.ask_count); "
                        "ctx._source.suggest = ['input': [params.text], 'weight': base + capped];"
                    ),
                    "params": {
                        "client_id": str(client_id),
                        "bot_id": str(bot_id),
                        "text": text,
                        "canonical_question": text,
                        "source": source_val,
                        "ts": _iso_from_ts(now_ts),
                        "updated_at": now_iso,
                        "base_weight": 8 if source_val == "user" else 4,
                    },
                },
                "upsert": payload,
            },
            refresh=False,
            request_timeout=ES_REQUEST_TIMEOUT,
        )
        log.debug(
            "ES question event indexed | client_id=%s | bot_id=%s | source=%s",
            client_id,
            bot_id,
            source_val,
        )
        return True
    except Exception as err:
        _mark_failure("index_question_event", err)
        return False


def search_question_suggestions(
    client_id: str,
    bot_id: str,
    query_text: str,
    *,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    query_text = _normalize_text(query_text)
    if not query_text:
        return []

    client = get_client()
    if not client:
        return []
    if not _ensure_question_index(client):
        return []

    out_by_text: dict[str, dict[str, Any]] = {}
    wanted = max(1, int(top_k))

    # Stage 1: prefix completion suggester (primary strategy).
    try:
        response = client.search(
            index=ES_QUESTION_INDEX,
            body={
                "_source": ["client_id", "bot_id", "text", "source"],
                "suggest": {
                    "question_prefix": {
                        "prefix": query_text,
                        "completion": {
                            "field": "suggest",
                            "size": max(20, wanted * 8),
                            "skip_duplicates": True,
                        },
                    }
                },
            },
            request_timeout=ES_REQUEST_TIMEOUT,
        )
        options = (
            (((response or {}).get("suggest") or {}).get("question_prefix") or [{}])[0].get("options", [])
        )
        for opt in options:
            source = opt.get("_source") or {}
            # Strict tenant/bot isolation after suggest retrieval.
            if str(source.get("client_id", "")) != str(client_id):
                continue
            if str(source.get("bot_id", "")) != str(bot_id):
                continue

            text = _normalize_text(str(opt.get("text") or source.get("text", "")))
            if not text:
                continue

            score = float(opt.get("_score", 0.0) or 0.0)
            prev = out_by_text.get(text)
            if prev is None or score > float(prev.get("score", 0.0) or 0.0):
                out_by_text[text] = {
                    "text": text,
                    "score": score,
                    "source": source.get("source", "seed"),
                }
    except Exception as err:
        _mark_failure("search_question_suggestions_prefix", err)

    # Stage 2: filtered lexical fallback for recall + recency shaping.
    try:
        response = client.search(
            index=ES_QUESTION_INDEX,
            body={
                "size": max(1, wanted * 2),
                "_source": ["text", "source"],
                "query": {
                    "function_score": {
                        "query": {
                            "bool": {
                                "filter": [
                                    {"term": {"client_id": str(client_id)}},
                                    {"term": {"bot_id": str(bot_id)}},
                                ],
                                "should": [
                                    {"match_phrase_prefix": {"text": {"query": query_text, "boost": 4.4}}},
                                    {"match_bool_prefix": {"text": {"query": query_text, "boost": 3.2}}},
                                    {"match": {"text": {"query": query_text, "operator": "and", "boost": 2.3}}},
                                ],
                                "minimum_should_match": 1,
                            }
                        },
                        "functions": [
                            {"filter": {"term": {"source": "user"}}, "weight": 1.25},
                            {
                                "gauss": {
                                    "ts": {
                                        "origin": "now",
                                        "scale": "21d",
                                        "offset": "1d",
                                        "decay": 0.6,
                                    }
                                }
                            },
                        ],
                        "score_mode": "sum",
                        "boost_mode": "sum",
                    }
                },
            },
            request_timeout=ES_REQUEST_TIMEOUT,
        )
        hits = (((response or {}).get("hits") or {}).get("hits") or [])
        for hit in hits:
            source = hit.get("_source") or {}
            text = _normalize_text(str(source.get("text", "")))
            if not text:
                continue
            score = float(hit.get("_score", 0.0) or 0.0)
            prev = out_by_text.get(text)
            if prev is None or score > float(prev.get("score", 0.0) or 0.0):
                out_by_text[text] = {
                    "text": text,
                    "score": score,
                    "source": source.get("source", "seed"),
                }
    except Exception as err:
        _mark_failure("search_question_suggestions_lexical", err)

    out = list(out_by_text.values())
    out.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    out = out[:wanted]

    log.debug(
        "ES autocomplete search | client_id=%s | bot_id=%s | query='%s' | hits=%d | top_k=%d | strategy=prefix_completion+lexical | took_ms=%.2f",
        client_id,
        bot_id,
        query_text[:72],
        len(out),
        wanted,
        (time.perf_counter() - start) * 1000.0,
    )
    return out
