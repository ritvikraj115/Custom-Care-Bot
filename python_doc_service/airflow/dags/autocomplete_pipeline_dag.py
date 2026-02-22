from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from urllib import error as url_error
from urllib import request as url_request

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _must_env(name: str) -> str:
    value = str(os.getenv(name, "")).strip()
    if not value:
        raise ValueError(f"Missing required env var: {name}")
    return value


def _service_base_url() -> str:
    return str(os.getenv("AIRFLOW_ML_SERVICE_URL", "http://host.docker.internal:8000")).strip().rstrip("/")


def _post_json(url: str, payload: dict, timeout_sec: int = 3600) -> dict:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = url_request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with url_request.urlopen(req, timeout=max(5, int(timeout_sec))) as resp:
            raw = (resp.read() or b"").decode("utf-8", errors="replace").strip()
            if not raw:
                return {}
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"raw": data}
    except url_error.HTTPError as err:
        detail = ""
        try:
            detail = (err.read() or b"").decode("utf-8", errors="replace")[:1000]
        except Exception:
            detail = str(err)
        raise RuntimeError(f"HTTP {err.code} for {url}: {detail}") from err
    except Exception as err:
        raise RuntimeError(f"Request failed for {url}: {err}") from err


def _dag_conf(context: dict, key: str) -> str:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None) or {}
    return str(conf.get(key, "")).strip()


def _dag_conf_bool(context: dict, key: str, default: bool = False) -> bool:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None) or {}
    raw = conf.get(key, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def autocomplete_retrain(**context) -> None:
    client_id = _dag_conf(context, "client_id") or _must_env("AIRFLOW_ML_CLIENT_ID")
    bot_id = _dag_conf(context, "bot_id") or _must_env("AIRFLOW_ML_BOT_ID")
    force = _dag_conf_bool(context, "force", default=False)
    service_url = _service_base_url()
    endpoint = f"{service_url}/autocomplete/train"
    response = _post_json(
        endpoint,
        {
            "client_id": str(client_id),
            "bot_id": str(bot_id),
            # Preserve model conditions: training runs only when existing logic says so.
            "wait": True,
            "force": bool(force),
        },
        timeout_sec=7200,
    )
    triggered = bool(response.get("triggered", False))
    status = response.get("status", {}) if isinstance(response, dict) else {}
    print(
        json.dumps(
            {
                "status": "ok",
                "service_url": service_url,
                "client_id": str(client_id),
                "bot_id": str(bot_id),
                "triggered": triggered,
                "force": bool(force),
                "pending_questions": status.get("pending_questions"),
                "model_version": status.get("model_version"),
                "last_error": status.get("last_error"),
            },
            ensure_ascii=True,
        )
    )
    if status.get("last_error"):
        raise RuntimeError(
            f"Autocomplete training failed | client_id={client_id} | bot_id={bot_id} | "
            f"error={status.get('last_error')}"
        )


def autocomplete_deploy() -> None:
    deploy_cmd = str(os.getenv("AIRFLOW_DEPLOY_CMD", "echo deploy_skipped")).strip()
    subprocess.run(deploy_cmd, cwd=str(PROJECT_ROOT), shell=True, check=True)


with DAG(
    dag_id="autocomplete_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "autocomplete", "production"],
) as dag:
    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=autocomplete_retrain,
    )

    deploy_task = PythonOperator(
        task_id="deploy_docker",
        python_callable=autocomplete_deploy,
    )

    retrain_task >> deploy_task
