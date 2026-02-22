from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT_DIR / "params.yaml"


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Update DVC params.yaml for dynamic bot runs.")
    parser.add_argument("--client-id", required=True, help="Client ID")
    parser.add_argument("--bot-id", required=True, help="Bot ID")
    parser.add_argument("--rebuild-mode", default="", help="Optional rebuild mode")
    parser.add_argument("--pdf-manifest", default="", help="Optional pdf manifest path")
    args = parser.parse_args()

    raw = ""
    if PARAMS_PATH.exists():
        raw = PARAMS_PATH.read_text(encoding="utf-8-sig")

    parsed = yaml.safe_load(raw) if raw.strip() else {}
    if not isinstance(parsed, dict):
        parsed = {}

    parsed = _clean_top_level_dvc_block(parsed)
    dvc_cfg = parsed.get("dvc", {})
    if not isinstance(dvc_cfg, dict):
        dvc_cfg = {}

    dvc_cfg["client_id"] = str(args.client_id)
    dvc_cfg["bot_id"] = str(args.bot_id)
    if str(args.rebuild_mode or "").strip():
        dvc_cfg["rebuild_mode"] = str(args.rebuild_mode)
    if str(args.pdf_manifest or "").strip():
        dvc_cfg["pdf_manifest"] = str(args.pdf_manifest)

    parsed["dvc"] = dvc_cfg

    dumped = yaml.safe_dump(
        parsed,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
    PARAMS_PATH.write_text(dumped, encoding="utf-8")
    print("params_updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
