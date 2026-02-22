from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

from app.pipeline.logger import get_logger

log = get_logger("airflow-trigger")


def _truthy(raw: str | None, default: bool = True) -> bool:
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def airflow_autotrigger_enabled() -> bool:
    return _truthy(os.getenv("AIRFLOW_AUTOTRIGGER_ENABLED"), default=True)


def _base_url() -> str:
    explicit = str(os.getenv("AIRFLOW_API_BASE_URL", "")).strip()
    if explicit:
        return explicit.rstrip("/")

    host = str(os.getenv("AIRFLOW_API_HOST", "")).strip()
    if host:
        port = str(os.getenv("AIRFLOW_API_PORT", "")).strip()
        if port:
            return f"http://{host}:{port}".rstrip("/")
        return f"http://{host}".rstrip("/")

    return "http://localhost:8081"


def _timeout_sec() -> int:
    raw = os.getenv("AIRFLOW_API_TIMEOUT_SEC", "20")
    try:
        return max(5, int(raw))
    except Exception:
        return 20


def _token_from_auth(base_url: str, timeout_sec: int) -> str:
    username = str(os.getenv("AIRFLOW_API_USERNAME", "airflow")).strip()
    password = str(os.getenv("AIRFLOW_API_PASSWORD", "airflow")).strip()

    payload = {"username": username, "password": password}
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = url_request.Request(
        f"{base_url}/auth/token",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with url_request.urlopen(req, timeout=max(5, int(timeout_sec))) as resp:
        raw = (resp.read() or b"").decode("utf-8", errors="replace").strip()
        data = json.loads(raw) if raw else {}
        token = str((data or {}).get("access_token", "")).strip()
        if not token:
            raise RuntimeError("Airflow token response missing access_token")
        return token


def _resolve_token(base_url: str, timeout_sec: int) -> str:
    env_token = str(os.getenv("AIRFLOW_API_TOKEN", "")).strip()
    if env_token:
        return env_token
    return _token_from_auth(base_url, timeout_sec)


def _trigger_dag_run(dag_id: str, conf: dict) -> tuple[bool, str]:
    base_url = _base_url()
    timeout_sec = _timeout_sec()
    max_attempts = 4
    last_error = "unknown_error"

    for attempt in range(1, max_attempts + 1):
        try:
            token = _resolve_token(base_url, timeout_sec=timeout_sec)
            logical_date = datetime.now(timezone.utc).isoformat()
            payload = {
                "logical_date": logical_date,
                "conf": conf,
            }
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            safe_dag = url_parse.quote(dag_id, safe="")
            endpoint = f"{base_url}/api/v2/dags/{safe_dag}/dagRuns"
            req = url_request.Request(
                endpoint,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                method="POST",
            )
            with url_request.urlopen(req, timeout=max(5, int(timeout_sec))) as resp:
                raw = (resp.read() or b"").decode("utf-8", errors="replace").strip()
                data = json.loads(raw) if raw else {}
                run_id = str((data or {}).get("dag_run_id", "")).strip()
                return True, run_id or "triggered"
        except url_error.HTTPError as err:
            detail = ""
            try:
                detail = (err.read() or b"").decode("utf-8", errors="replace")[:400]
            except Exception:
                detail = str(err)
            last_error = f"http_{err.code}:{detail}"
            if err.code >= 500 and attempt < max_attempts:
                time.sleep(0.4 * attempt)
                continue
            return False, last_error
        except Exception as err:
            last_error = str(err)
            if attempt < max_attempts:
                time.sleep(0.4 * attempt)
                continue
            return False, last_error

    return False, last_error


def trigger_autocomplete_retrain_dag(
    client_id: str,
    bot_id: str,
    reason: str = "retrain_condition_met",
    force: bool = False,
) -> tuple[bool, str]:
    if not airflow_autotrigger_enabled():
        return False, "airflow_autotrigger_disabled"

    dag_id = str(os.getenv("AIRFLOW_AUTOCOMPLETE_DAG_ID", "autocomplete_pipeline")).strip()
    dag_id = dag_id or "autocomplete_pipeline"
    return _trigger_dag_run(
        dag_id,
        conf={
            "client_id": str(client_id),
            "bot_id": str(bot_id),
            "reason": str(reason),
            "force": bool(force),
        },
    )


def trigger_hdbscan_cluster_dag(
    client_id: str,
    bot_id: str,
    rebuild_mode: str = "full",
    pdf_manifest: str = "data/dvc/pdf_manifest.json",
    reason: str = "frontend_pdf_upload",
    skip_train: bool = False,
) -> tuple[bool, str]:
    if not airflow_autotrigger_enabled():
        return False, "airflow_autotrigger_disabled"

    dag_id = str(os.getenv("AIRFLOW_HDBSCAN_DAG_ID", "hdbscan_pipeline")).strip()
    dag_id = dag_id or "hdbscan_pipeline"
    return _trigger_dag_run(
        dag_id,
        conf={
            "client_id": str(client_id),
            "bot_id": str(bot_id),
            "rebuild_mode": str(rebuild_mode or "full"),
            "pdf_manifest": str(pdf_manifest or "data/dvc/pdf_manifest.json"),
            "reason": str(reason),
            "skip_train": bool(skip_train),
        },
    )
