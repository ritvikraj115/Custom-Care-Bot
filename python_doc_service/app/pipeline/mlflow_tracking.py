from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from app.pipeline.logger import get_logger
from app.pipeline.storage import BASE_DIR, ensure_dir

log = get_logger("mlflow")

try:
    import mlflow
except Exception as err:
    mlflow = None
    _MLFLOW_IMPORT_ERR = err
else:
    _MLFLOW_IMPORT_ERR = None

_MLFLOW_CONFIGURED = False
_MLFLOW_DISABLED_LOGGED = False


def _truthy(raw: str | None, default: bool = True) -> bool:
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_enabled() -> bool:
    return bool(mlflow is not None) and _truthy(os.getenv("MLFLOW_ENABLED"), default=True)


def _experiment_name(component: str) -> str:
    env_key = f"MLFLOW_EXPERIMENT_{str(component).strip().upper()}"
    fallback = f"bot-{str(component).strip().lower()}"
    name = str(os.getenv(env_key, fallback)).strip()
    return name or fallback


def _is_windows_drive_path(raw: str) -> bool:
    return len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha()


def _normalize_tracking_uri(raw_uri: str) -> str:
    raw = str(raw_uri or "").strip()
    if not raw:
        return ""

    # On Windows, paths like "C:\..." are parsed as an unsupported URI scheme ("c").
    # Convert any local filesystem path to an explicit file URI.
    if _is_windows_drive_path(raw):
        path = Path(raw).expanduser().resolve()
        ensure_dir(str(path))
        return path.as_uri()

    parsed = urlparse(raw)
    if not parsed.scheme:
        path = Path(raw).expanduser().resolve()
        ensure_dir(str(path))
        return path.as_uri()

    return raw


def _ensure_tracking_configured() -> bool:
    global _MLFLOW_CONFIGURED
    global _MLFLOW_DISABLED_LOGGED

    if not _is_enabled():
        if not _MLFLOW_DISABLED_LOGGED:
            if mlflow is None and _MLFLOW_IMPORT_ERR is not None:
                log.warning("MLflow disabled: import failed | err=%s", _MLFLOW_IMPORT_ERR)
            else:
                log.info("MLflow disabled via MLFLOW_ENABLED")
            _MLFLOW_DISABLED_LOGGED = True
        return False

    if _MLFLOW_CONFIGURED:
        return True

    try:
        tracking_uri = str(os.getenv("MLFLOW_TRACKING_URI", "")).strip()
        if not tracking_uri:
            tracking_host = str(os.getenv("MLFLOW_TRACKING_HOST", "")).strip()
            tracking_port = str(os.getenv("MLFLOW_TRACKING_PORT", "")).strip()
            if tracking_host:
                if tracking_port:
                    tracking_uri = f"http://{tracking_host}:{tracking_port}"
                else:
                    tracking_uri = f"http://{tracking_host}"
        if not tracking_uri:
            tracking_path = Path(BASE_DIR).resolve() / "mlruns"
            ensure_dir(str(tracking_path))
            tracking_uri = tracking_path.as_uri()
        else:
            tracking_uri = _normalize_tracking_uri(tracking_uri)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        _MLFLOW_CONFIGURED = True
        log.info("MLflow configured | tracking_uri=%s", tracking_uri)
        return True
    except Exception as err:
        log.warning("MLflow setup failed | err=%s", err)
        return False


def _safe_param_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


@contextmanager
def start_bot_run(
    component: str,
    client_id: str,
    bot_id: str,
    run_name: str | None = None,
    extra_tags: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    if not _ensure_tracking_configured():
        yield None
        return

    tags = {
        "client_id": str(client_id),
        "bot_id": str(bot_id),
        "component": str(component),
    }
    if extra_tags:
        tags.update({str(k): str(v) for k, v in extra_tags.items() if v is not None})

    try:
        mlflow.set_experiment(_experiment_name(component))
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tags(tags)
            yield mlflow
    except Exception as err:
        log.warning(
            "MLflow run failed to start | component=%s | client_id=%s | bot_id=%s | err=%s",
            component,
            client_id,
            bot_id,
            err,
        )
        yield None


def set_tags(tracker: Any | None, tags: dict[str, Any] | None) -> None:
    if tracker is None or not tags:
        return
    payload = {str(k): str(v) for k, v in tags.items() if v is not None}
    if not payload:
        return
    try:
        tracker.set_tags(payload)
    except Exception as err:
        log.warning("MLflow set_tags failed | err=%s", err)


def log_params(tracker: Any | None, params: dict[str, Any] | None) -> None:
    if tracker is None or not params:
        return
    payload = {}
    for key, value in params.items():
        safe_val = _safe_param_value(value)
        if safe_val is not None:
            payload[str(key)] = safe_val
    if not payload:
        return
    try:
        tracker.log_params(payload)
    except Exception as err:
        log.warning("MLflow log_params failed | err=%s", err)


def log_metrics(
    tracker: Any | None,
    metrics: dict[str, Any] | None,
    step: int | None = None,
) -> None:
    if tracker is None or not metrics:
        return
    payload = {}
    for key, value in metrics.items():
        try:
            payload[str(key)] = float(value)
        except Exception:
            continue
    if not payload:
        return
    try:
        if step is None:
            tracker.log_metrics(payload)
        else:
            tracker.log_metrics(payload, step=step)
    except Exception as err:
        log.warning("MLflow log_metrics failed | err=%s", err)


def log_dict(tracker: Any | None, data: Any, artifact_file: str) -> None:
    if tracker is None:
        return
    try:
        tracker.log_dict(data, artifact_file)
    except Exception as err:
        log.warning("MLflow log_dict failed | artifact=%s | err=%s", artifact_file, err)


def log_artifact(
    tracker: Any | None,
    local_path: str | None,
    artifact_path: str | None = None,
) -> None:
    if tracker is None or not local_path:
        return
    if not os.path.exists(local_path):
        return
    try:
        if artifact_path:
            tracker.log_artifact(local_path, artifact_path=artifact_path)
        else:
            tracker.log_artifact(local_path)
    except Exception as err:
        log.warning("MLflow log_artifact failed | path=%s | err=%s", local_path, err)
