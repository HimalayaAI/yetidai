"""
core/request_log.py — append-only JSONL telemetry for YetiDai turns.

One line per user turn written to logs/yeti.jsonl. Tail with:
    tail -F logs/yeti.jsonl | jq .
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOG_DIR = Path(os.getenv("YETI_LOG_DIR", "logs"))
_LOG_FILE = _LOG_DIR / "yeti.jsonl"
_LOCK = threading.Lock()


def _ensure_dir() -> None:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def log_turn(**fields: Any) -> None:
    """Append one JSON line. Never raises — logging must not break the bot."""
    try:
        _ensure_dir()
        record = {"ts": time.time(), **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOCK, _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
