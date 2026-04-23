"""Append-only JSONL telemetry log. One line per event. Safe to call from any
thread; small writes are serialized by a module-level lock."""
import json
import os
import threading
import time

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetry.jsonl")
_lock = threading.Lock()


def reset():
    with _lock:
        try:
            open(LOG_PATH, "w").close()
        except Exception:
            pass


def log(src, **kwargs):
    event = {"t": time.time(), "src": src}
    event.update(kwargs)
    line = json.dumps(event, default=_default)
    with _lock:
        try:
            with open(LOG_PATH, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


def _default(obj):
    try:
        return float(obj)
    except Exception:
        return str(obj)
