"""
System Status Router (live)
============================
GET /api/system-status — real registry state (loaded, checkpoint, device,
trained_at, headline test metric), library versions, the age of
incidents.json and whether weather is configured. No invented percentages.
"""

import json
import platform
from datetime import datetime
from importlib import metadata

from fastapi import APIRouter

import config
from ai import registry

router = APIRouter()

PACKAGES = ["torch", "ultralytics", "transformers", "fastapi"]


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def _incidents_info() -> dict:
    path = config.INCIDENTS_PATH
    info = {"path": path.relative_to(config.REPO_ROOT).as_posix(), "exists": path.exists(),
            "generated_at": None, "count": 0, "age_minutes": None}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            info["generated_at"] = payload.get("generated_at")
            info["count"] = len(payload.get("incidents", []))
            generated = datetime.fromisoformat(payload["generated_at"])
            info["age_minutes"] = round((datetime.now(generated.tzinfo) - generated).total_seconds() / 60, 1)
        except Exception as exc:
            info["error"] = f"could not parse: {exc}"
    return info


@router.get("", summary="Real system state")
def get_system_status():
    try:
        device = config.resolve_device()
    except RuntimeError as exc:
        device = f"error: {exc}"
    return {
        "data_origin": "live",
        "models": {"vision": registry.VISION.public_status(), "nlp": registry.NLP.public_status()},
        "device": device,
        "versions": {"python": platform.python_version(), **{p: _version(p) for p in PACKAGES}},
        "incidents": _incidents_info(),
        "weather": {"configured": bool(config.OPENWEATHER_KEY)},
        "llm_second_opinion": {
            "configured": bool(config.GROQ_API_KEY),
            "provider": "Groq",
            "model": config.GROQ_VISION_MODEL,
            "used_in_fusion": False,
        },
    }
