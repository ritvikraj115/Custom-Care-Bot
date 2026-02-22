from __future__ import annotations

import os
import json
import subprocess
import sys
import threading
from pathlib import Path

import yaml

from app.pipeline.logger import get_logger

log = get_logger("dvc-auto")

_GLOBAL_LOCK = threading.Lock()


def _truthy(raw: str | None, default: bool = True) -> bool:
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_enabled() -> bool:
    return _truthy(os.getenv("DVC_AUTO_FLOW_ENABLED"), default=True)


def _is_push_enabled() -> bool:
    return _truthy(os.getenv("DVC_AUTO_PUSH"), default=False)


def _timeout_sec() -> int:
    raw = os.getenv("DVC_AUTO_TIMEOUT_SEC", "900")
    try:
        return max(30, int(raw))
    except Exception:
        return 900


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clean_top_level_dvc_block(data: dict) -> dict:
    merged_dvc: dict = {}
    cleaned: dict = {}
    for key, value in (data or {}).items():
        normalized_key = str(key).replace("\ufeff", "")
        if normalized_key == "dvc":
            if isinstance(value, dict):
                merged_dvc.update(value)
            continue
        cleaned[str(key)] = value
    cleaned["dvc"] = merged_dvc
    return cleaned


def _set_dvc_params(root: Path, client_id: str, bot_id: str) -> None:
    params_path = root / "params.yaml"
    raw = ""
    if params_path.exists():
        raw = params_path.read_text(encoding="utf-8-sig")

    parsed = yaml.safe_load(raw) if raw.strip() else {}
    if not isinstance(parsed, dict):
        parsed = {}

    parsed = _clean_top_level_dvc_block(parsed)
    dvc_cfg = parsed.get("dvc", {})
    if not isinstance(dvc_cfg, dict):
        dvc_cfg = {}
    dvc_cfg["client_id"] = str(client_id)
    dvc_cfg["bot_id"] = str(bot_id)
    parsed["dvc"] = dvc_cfg

    dumped = yaml.safe_dump(
        parsed,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
    params_path.write_text(dumped, encoding="utf-8")


def _set_dvc_params_extended(
    root: Path,
    client_id: str,
    bot_id: str,
    rebuild_mode: str = "full",
    pdf_manifest: str = "data/dvc/pdf_manifest.json",
) -> None:
    params_path = root / "params.yaml"
    raw = ""
    if params_path.exists():
        raw = params_path.read_text(encoding="utf-8-sig")

    parsed = yaml.safe_load(raw) if raw.strip() else {}
    if not isinstance(parsed, dict):
        parsed = {}

    parsed = _clean_top_level_dvc_block(parsed)
    dvc_cfg = parsed.get("dvc", {})
    if not isinstance(dvc_cfg, dict):
        dvc_cfg = {}

    dvc_cfg["client_id"] = str(client_id)
    dvc_cfg["bot_id"] = str(bot_id)
    dvc_cfg["rebuild_mode"] = str(rebuild_mode or "full")
    dvc_cfg["pdf_manifest"] = str(pdf_manifest or "data/dvc/pdf_manifest.json")
    parsed["dvc"] = dvc_cfg

    dumped = yaml.safe_dump(
        parsed,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
    params_path.write_text(dumped, encoding="utf-8")


def _run_dvc_command(args: list[str], cwd: Path, timeout_sec: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        timeout=max(30, int(timeout_sec)),
    )
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def track_autocomplete_training(client_id: str, bot_id: str, model_version: int | None = None) -> None:
    if not _is_enabled():
        return

    root = _project_root()
    timeout_sec = _timeout_sec()

    with _GLOBAL_LOCK:
        _set_dvc_params(root, client_id=client_id, bot_id=bot_id)
        cmd = [
            sys.executable,
            "-m",
            "dvc",
            "repro",
            "autocomplete_track",
        ]
        rc, out, err = _run_dvc_command(cmd, cwd=root, timeout_sec=timeout_sec)
        if rc != 0:
            log.warning(
                "DVC autocomplete tracking failed | client_id=%s | bot_id=%s | model_version=%s | rc=%s | err=%s",
                client_id,
                bot_id,
                model_version,
                rc,
                (err or out)[:1000],
            )
            return

        log.info(
            "DVC autocomplete tracking complete | client_id=%s | bot_id=%s | model_version=%s",
            client_id,
            bot_id,
            model_version,
        )

        if not _is_push_enabled():
            return

        push_cmd = [sys.executable, "-m", "dvc", "push"]
        prc, pout, perr = _run_dvc_command(push_cmd, cwd=root, timeout_sec=timeout_sec)
        if prc != 0:
            log.warning(
                "DVC push failed | client_id=%s | bot_id=%s | rc=%s | err=%s",
                client_id,
                bot_id,
                prc,
                (perr or pout)[:1000],
            )
        else:
            log.info(
                "DVC push complete | client_id=%s | bot_id=%s",
                client_id,
                bot_id,
            )


def run_cluster_training(
    client_id: str,
    bot_id: str,
    rebuild_mode: str = "full",
    pdf_manifest: str = "data/dvc/pdf_manifest.json",
) -> dict:
    if not _is_enabled():
        return {
            "ok": False,
            "error": "dvc_auto_flow_disabled",
        }

    root = _project_root()
    timeout_sec = _timeout_sec()

    summary_rel = (
        f"storage/dvc_runs/client_{client_id}/bot_{bot_id}/clustering_summary.json"
    )
    summary_path = root / summary_rel

    with _GLOBAL_LOCK:
        _set_dvc_params_extended(
            root,
            client_id=client_id,
            bot_id=bot_id,
            rebuild_mode=rebuild_mode,
            pdf_manifest=pdf_manifest,
        )

        cmd = [
            sys.executable,
            "-m",
            "dvc",
            "repro",
            "cluster_train",
        ]
        rc, out, err = _run_dvc_command(cmd, cwd=root, timeout_sec=timeout_sec)
        if rc != 0:
            return {
                "ok": False,
                "error": f"dvc_cluster_repro_failed: {(err or out)[:1000]}",
            }

        payload = {}
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}

        if _is_push_enabled():
            push_cmd = [sys.executable, "-m", "dvc", "push"]
            _run_dvc_command(push_cmd, cwd=root, timeout_sec=timeout_sec)

        return {
            "ok": True,
            "summary_file": str(summary_path),
            "summary": payload.get("summary", {}) if isinstance(payload, dict) else {},
            "raw": payload if isinstance(payload, dict) else {},
        }
