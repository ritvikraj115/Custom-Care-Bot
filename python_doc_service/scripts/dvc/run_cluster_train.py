from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except Exception:
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_existing_path(raw_path: str, manifest_dir: Path) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        raise ValueError("Empty PDF path in manifest")

    candidates = []
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((manifest_dir / p).resolve())
        candidates.append((ROOT_DIR / p).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    joined = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"PDF not found: '{raw}'. Checked: {joined}")


def _parse_manifest(manifest_path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("pdfs", [])
    else:
        raise ValueError("PDF manifest must be a list or an object with 'pdfs'")

    if not isinstance(entries, list):
        raise ValueError("'pdfs' must be a list")
    if not entries:
        raise ValueError("No PDF entries found in manifest")

    manifest_dir = manifest_path.parent
    pdf_paths: list[str] = []
    pdf_metadata: dict[str, dict[str, Any]] = {}

    for entry in entries:
        if isinstance(entry, str):
            raw_path = entry
            source_type = None
            source_url = None
        elif isinstance(entry, dict):
            raw_path = entry.get("path", "")
            source_type = entry.get("source_type")
            source_url = entry.get("source_url")
        else:
            raise ValueError("Each PDF entry must be a string or an object")

        resolved = _resolve_existing_path(str(raw_path), manifest_dir)
        pdf_paths.append(str(resolved))

        meta: dict[str, Any] = {}
        if source_type:
            meta["source_type"] = str(source_type).strip().lower()
        if source_url:
            meta["source_url"] = str(source_url).strip()
        if meta:
            pdf_metadata[resolved.name] = meta

    return pdf_paths, pdf_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bot-specific clustering pipeline for DVC.")
    parser.add_argument("--client-id", required=True, help="Client ID")
    parser.add_argument("--bot-id", required=True, help="Bot ID")
    parser.add_argument("--pdf-manifest", required=True, help="Path to JSON manifest of PDFs")
    parser.add_argument("--summary-out", required=True, help="Output JSON summary path")
    parser.add_argument("--rebuild-mode", default="full", help="full or incremental")
    args = parser.parse_args()

    from app.pipeline.run_pipeline import run_pipeline
    from app.pipeline.storage import ensure_dir

    manifest_path = Path(args.pdf_manifest)
    if not manifest_path.is_absolute():
        manifest_path = (ROOT_DIR / manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    pdf_paths, pdf_metadata = _parse_manifest(manifest_path)
    summary = run_pipeline(
        pdf_paths=pdf_paths,
        bot_id=str(args.bot_id),
        client_id=str(args.client_id),
        pdf_metadata=pdf_metadata or None,
        rebuild_mode=str(args.rebuild_mode or "full"),
    )

    out_path = Path(args.summary_out)
    if not out_path.is_absolute():
        out_path = (ROOT_DIR / out_path).resolve()
    ensure_dir(str(out_path.parent))
    payload = {
        "generated_at": _utc_now_iso(),
        "client_id": str(args.client_id),
        "bot_id": str(args.bot_id),
        "rebuild_mode": str(args.rebuild_mode or "full"),
        "pdf_manifest": str(manifest_path),
        "pdf_count": len(pdf_paths),
        "summary": summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    print(json.dumps({"status": "ok", "summary_out": str(out_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
