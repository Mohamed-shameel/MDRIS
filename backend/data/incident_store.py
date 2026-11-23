"""
Simulated Incident Store
==========================
Reads backend/data/incidents.json, written offline by training/generate_incidents.py
(docs/MDRIS_CONTEXT.md section 10). Every incident is simulated: real held-out
dataset inputs run through the real models, with synthetic places and times.

If the file is missing the incident endpoints return HTTP 503 with the reason;
nothing is invented to fill the gap.
"""

import json
import logging
import threading
from typing import Any, Dict, List

import config

logger = logging.getLogger("mdris.data.incidents")


class IncidentsUnavailable(RuntimeError):
    """incidents.json is missing or unreadable (-> HTTP 503)."""


_lock = threading.Lock()
_cache: Dict[str, Any] = {"mtime": None, "payload": None}


def load() -> Dict[str, Any]:
    """Return the parsed incidents.json, re-reading it when the file changes."""
    path = config.INCIDENTS_PATH
    if not path.exists():
        raise IncidentsUnavailable(
            f"simulated incidents not generated: {path.name} not found (run training/generate_incidents.py)"
        )
    mtime = path.stat().st_mtime
    with _lock:
        if _cache["mtime"] != mtime:
            try:
                _cache["payload"] = json.loads(path.read_text(encoding="utf-8"))
                _cache["mtime"] = mtime
            except Exception as exc:
                logger.error(f"Could not parse {path}: {exc}")
                raise IncidentsUnavailable(f"could not parse {path.name}: {exc}") from exc
        return _cache["payload"]


def sorted_incidents() -> List[Dict[str, Any]]:
    """All incidents, priority descending, then evidence strength descending."""
    incidents = load()["incidents"]
    return sorted(
        incidents,
        key=lambda i: (-i["fusion"]["priority_score"], -i["fusion"]["evidence_strength"], i["id"]),
    )
