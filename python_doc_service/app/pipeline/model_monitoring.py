from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from app.pipeline.logger import get_logger

log = get_logger("monitoring")

BASE_DIR = "storage"
MONITORING_ROOT = os.path.join(BASE_DIR, "monitoring")
DASHBOARD_FILE = os.path.join(MONITORING_ROOT, "model_dashboard.json")

_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truthy(raw: str | None, default: bool = True) -> bool:
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _enabled() -> bool:
    return _truthy(os.getenv("MODEL_MONITORING_ENABLED"), default=True)


def _hdbscan_drift_threshold() -> float:
    try:
        return max(0.01, min(1.0, float(os.getenv("HDBSCAN_DRIFT_THRESHOLD", "0.35"))))
    except Exception:
        return 0.35


def _autocomplete_drift_threshold() -> float:
    try:
        return max(0.01, min(1.0, float(os.getenv("AUTOCOMPLETE_DRIFT_THRESHOLD", "0.35"))))
    except Exception:
        return 0.35


def _autocomplete_min_confidence() -> float:
    try:
        return max(0.0, min(1.0, float(os.getenv("AUTOCOMPLETE_MIN_CONFIDENCE", "0.35"))))
    except Exception:
        return 0.35


def _safe_bucket(client_id: str, bot_id: str) -> str:
    return f"{str(client_id)}::{str(bot_id)}"


def _default_dashboard() -> dict[str, Any]:
    return {
        "updated_at": _utc_now_iso(),
        "hdbscan": {"global": {}, "bots": {}},
        "autocomplete": {"global": {}, "bots": {}},
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_dashboard() -> dict[str, Any]:
    if not os.path.exists(DASHBOARD_FILE):
        return _default_dashboard()
    try:
        with open(DASHBOARD_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload.setdefault("hdbscan", {"global": {}, "bots": {}})
            payload.setdefault("autocomplete", {"global": {}, "bots": {}})
            payload["hdbscan"].setdefault("global", {})
            payload["hdbscan"].setdefault("bots", {})
            payload["autocomplete"].setdefault("global", {})
            payload["autocomplete"].setdefault("bots", {})
            payload.setdefault("updated_at", _utc_now_iso())
            return payload
    except Exception:
        pass
    return _default_dashboard()


def _save_dashboard(payload: dict[str, Any]) -> None:
    _ensure_dir(MONITORING_ROOT)
    payload["updated_at"] = _utc_now_iso()
    with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _init_hdbscan_stats() -> dict[str, Any]:
    return {
        "queries_total": 0,
        "uncategorized_queries": 0,
        "semantic_empty_queries": 0,
        "last_event_at": None,
        "last_reason": None,
        "drift_rate": 0.0,
        "drift_alert": False,
    }


def _init_autocomplete_stats() -> dict[str, Any]:
    return {
        "queries_total": 0,
        "empty_suggestions": 0,
        "low_confidence_queries": 0,
        "drift_events": 0,
        "avg_max_confidence": 0.0,
        "last_event_at": None,
        "last_reason": None,
        "drift_rate": 0.0,
        "drift_alert": False,
    }


def _hdbscan_update(
    stats: dict[str, Any],
    uncategorized: bool,
    semantic_empty: bool,
    reason: str,
) -> None:
    stats["queries_total"] = int(stats.get("queries_total", 0) or 0) + 1
    if uncategorized:
        stats["uncategorized_queries"] = int(stats.get("uncategorized_queries", 0) or 0) + 1
    if semantic_empty:
        stats["semantic_empty_queries"] = int(stats.get("semantic_empty_queries", 0) or 0) + 1

    total = max(1, int(stats["queries_total"]))
    uncategorized_count = int(stats.get("uncategorized_queries", 0) or 0)
    drift_rate = float(uncategorized_count / total)
    threshold = _hdbscan_drift_threshold()

    stats["drift_rate"] = round(drift_rate, 4)
    stats["drift_alert"] = bool(total >= 20 and drift_rate >= threshold)
    stats["last_event_at"] = _utc_now_iso()
    stats["last_reason"] = str(reason)


def _autocomplete_update(
    stats: dict[str, Any],
    empty_suggestions: bool,
    low_confidence: bool,
    max_confidence: float,
    reason: str,
) -> None:
    stats["queries_total"] = int(stats.get("queries_total", 0) or 0) + 1
    if empty_suggestions:
        stats["empty_suggestions"] = int(stats.get("empty_suggestions", 0) or 0) + 1
    if low_confidence:
        stats["low_confidence_queries"] = int(stats.get("low_confidence_queries", 0) or 0) + 1

    total = max(1, int(stats["queries_total"]))
    drift_events = int(stats.get("empty_suggestions", 0) or 0) + int(stats.get("low_confidence_queries", 0) or 0)
    stats["drift_events"] = int(drift_events)

    prev_avg = float(stats.get("avg_max_confidence", 0.0) or 0.0)
    stats["avg_max_confidence"] = round(((prev_avg * (total - 1)) + float(max_confidence)) / total, 4)

    drift_rate = float(drift_events / total)
    threshold = _autocomplete_drift_threshold()
    stats["drift_rate"] = round(drift_rate, 4)
    stats["drift_alert"] = bool(total >= 20 and drift_rate >= threshold)
    stats["last_event_at"] = _utc_now_iso()
    stats["last_reason"] = str(reason)


def record_hdbscan_query_event(
    client_id: str,
    bot_id: str,
    final_hits: int,
    semantic_hits: int,
    candidate_clusters: int,
    reason: str,
) -> None:
    if not _enabled():
        return

    uncategorized = int(final_hits) <= 0
    semantic_empty = int(semantic_hits) <= 0
    bucket = _safe_bucket(client_id, bot_id)

    with _LOCK:
        dashboard = _load_dashboard()
        model_section = dashboard.setdefault("hdbscan", {"global": {}, "bots": {}})
        bot_stats = model_section.setdefault("bots", {}).get(bucket) or _init_hdbscan_stats()
        global_stats = model_section.setdefault("global", {}) or _init_hdbscan_stats()

        _hdbscan_update(bot_stats, uncategorized=uncategorized, semantic_empty=semantic_empty, reason=reason)
        _hdbscan_update(global_stats, uncategorized=uncategorized, semantic_empty=semantic_empty, reason=reason)

        model_section["bots"][bucket] = bot_stats
        model_section["global"] = global_stats
        _save_dashboard(dashboard)

    if uncategorized and bool(bot_stats.get("drift_alert")):
        log.warning(
            "HDBSCAN drift alert | client_id=%s | bot_id=%s | drift_rate=%.3f | reason=%s | clusters=%d",
            client_id,
            bot_id,
            float(bot_stats.get("drift_rate", 0.0) or 0.0),
            reason,
            int(candidate_clusters),
        )


def record_autocomplete_query_event(
    client_id: str,
    bot_id: str,
    suggestions: list[dict[str, Any]] | None,
    reason: str,
) -> None:
    if not _enabled():
        return

    rows = suggestions or []
    max_confidence = 0.0
    for row in rows:
        try:
            max_confidence = max(max_confidence, float(row.get("confidence", 0.0) or 0.0))
        except Exception:
            continue

    empty_suggestions = len(rows) == 0
    low_confidence = (not empty_suggestions) and (max_confidence < _autocomplete_min_confidence())
    bucket = _safe_bucket(client_id, bot_id)

    with _LOCK:
        dashboard = _load_dashboard()
        model_section = dashboard.setdefault("autocomplete", {"global": {}, "bots": {}})
        bot_stats = model_section.setdefault("bots", {}).get(bucket) or _init_autocomplete_stats()
        global_stats = model_section.setdefault("global", {}) or _init_autocomplete_stats()

        _autocomplete_update(
            bot_stats,
            empty_suggestions=empty_suggestions,
            low_confidence=low_confidence,
            max_confidence=max_confidence,
            reason=reason,
        )
        _autocomplete_update(
            global_stats,
            empty_suggestions=empty_suggestions,
            low_confidence=low_confidence,
            max_confidence=max_confidence,
            reason=reason,
        )

        model_section["bots"][bucket] = bot_stats
        model_section["global"] = global_stats
        _save_dashboard(dashboard)

    if (empty_suggestions or low_confidence) and bool(bot_stats.get("drift_alert")):
        log.warning(
            "Autocomplete drift alert | client_id=%s | bot_id=%s | drift_rate=%.3f | reason=%s | max_conf=%.3f",
            client_id,
            bot_id,
            float(bot_stats.get("drift_rate", 0.0) or 0.0),
            reason,
            float(max_confidence),
        )


def get_model_dashboard() -> dict[str, Any]:
    with _LOCK:
        return _load_dashboard()
