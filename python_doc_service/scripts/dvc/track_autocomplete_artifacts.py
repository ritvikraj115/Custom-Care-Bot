from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")
ROOT_DIR = Path(__file__).resolve().parents[2]
STORAGE_ROOT = ROOT_DIR / "storage" / "autocomplete"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(raw: str) -> str:
    return SAFE_ID_RE.sub("_", str(raw or "unknown")).strip("_") or "unknown"


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "mtime": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write bot autocomplete artifact tracking summary for DVC.")
    parser.add_argument("--client-id", required=True, help="Client ID")
    parser.add_argument("--bot-id", required=True, help="Bot ID")
    parser.add_argument("--summary-out", required=True, help="Output JSON summary path")
    args = parser.parse_args()

    safe_client = _safe_id(args.client_id)
    safe_bot = _safe_id(args.bot_id)
    bot_dir = STORAGE_ROOT / f"client_{safe_client}" / f"bot_{safe_bot}"

    state_path = bot_dir / "state.json"
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}
    else:
        state = {}

    payload = {
        "generated_at": _utc_now_iso(),
        "client_id": str(args.client_id),
        "bot_id": str(args.bot_id),
        "safe_client_id": safe_client,
        "safe_bot_id": safe_bot,
        "bot_dir": str(bot_dir),
        "state": state if isinstance(state, dict) else {},
        "artifacts": {
            "model": _file_info(bot_dir / "model.keras"),
            "weights": _file_info(bot_dir / "model.weights.h5"),
            "tokenizer_model": _file_info(bot_dir / "tokenizer.model"),
            "tokenizer_vocab": _file_info(bot_dir / "tokenizer.vocab"),
            "questions": _file_info(bot_dir / "questions.jsonl"),
            "state": _file_info(state_path),
        },
    }

    out_path = Path(args.summary_out)
    if not out_path.is_absolute():
        out_path = (ROOT_DIR / out_path).resolve()
    os.makedirs(str(out_path.parent), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    print(json.dumps({"status": "ok", "summary_out": str(out_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
