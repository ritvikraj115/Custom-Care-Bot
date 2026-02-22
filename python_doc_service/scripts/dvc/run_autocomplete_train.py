from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bot-specific autocomplete training for DVC.")
    parser.add_argument("--client-id", required=True, help="Client ID")
    parser.add_argument("--bot-id", required=True, help="Bot ID")
    parser.add_argument("--summary-out", required=True, help="Output JSON summary path")
    parser.add_argument("--wait", action="store_true", help="Wait for training thread to finish")
    parser.add_argument("--force", action="store_true", help="Force train regardless of pending thresholds")
    args = parser.parse_args()

    # Avoid nested DVC calls while this script itself is running as a DVC stage.
    os.environ["DVC_AUTO_FLOW_ENABLED"] = "false"

    from app.autocomplete_training_pipeline import get_autocomplete_manager
    from app.pipeline.storage import ensure_dir

    manager = get_autocomplete_manager()
    status_before = manager.get_status(str(args.client_id), str(args.bot_id))
    triggered = manager.trigger_training(
        client_id=str(args.client_id),
        bot_id=str(args.bot_id),
        wait=bool(args.wait),
        force=bool(args.force),
    )
    status_after = manager.get_status(str(args.client_id), str(args.bot_id))

    out_path = Path(args.summary_out)
    if not out_path.is_absolute():
        out_path = (ROOT_DIR / out_path).resolve()
    ensure_dir(str(out_path.parent))

    payload = {
        "generated_at": _utc_now_iso(),
        "client_id": str(args.client_id),
        "bot_id": str(args.bot_id),
        "wait": bool(args.wait),
        "force": bool(args.force),
        "triggered": bool(triggered),
        "status_before": status_before,
        "status_after": status_after,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    print(json.dumps({"status": "ok", "triggered": bool(triggered), "summary_out": str(out_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
