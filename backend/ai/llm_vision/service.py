"""
LLM Vision — Second Opinion (Groq)
====================================
Sends an uploaded flood image to a Groq-hosted vision LLM and asks it to
list the people and vehicles it sees, using the four canonical classes
(docs/MDRIS_CONTEXT.md D-36).

This is a SECOND OPINION shown beside the YOLOv8n result:
  - it is never an input to fusion,
  - it is always labelled with the LLM's name (data_origin "external_llm"),
  - it reports counts and descriptions only; an LLM gives no calibrated
    per-object confidence, so none is invented.

Without GROQ_API_KEY the service raises LlmNotConfigured (-> HTTP 503).
A failed request or an invalid reply raises LlmUnavailable (-> HTTP 502).
"""

import base64
import io
import json
import logging
import time
from typing import Any, Dict, List, Literal

import httpx
from pydantic import BaseModel, Field

import config
from ai.fusion.constants import CLASS_NAMES

logger = logging.getLogger("mdris.llm_vision")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PROVIDER = "Groq"
MAX_SIDE = 1024          # downscale before sending (Groq caps base64 images at 4 MB)
TIMEOUT_S = 45.0

PROMPT = f"""You are checking a flood-response image for people and vehicles in danger.
List every person and every vehicle you can see. Put each one in exactly one class:
- person_in_water: a person standing, wading, swimming or stranded in flood water
- person_in_vehicle: a person inside or on top of a vehicle that is in flood water
- person: any other visible person (dry ground, rooftop, boat, balcony)
- vehicle: a car, bus, truck, motorbike or auto-rickshaw with no visible occupant
Count only what is visible. Do not guess at hidden people.
Reply with JSON only, in this exact shape:
{{"flood_visible": true, "objects": [{{"class_name": "person_in_water", "description": "woman carrying a basin, knee-deep water"}}], "summary": "one sentence"}}
class_name must be one of {CLASS_NAMES}."""


class LlmNotConfigured(RuntimeError):
    """GROQ_API_KEY is not set (-> HTTP 503)."""


class LlmUnavailable(RuntimeError):
    """The Groq request failed or returned an invalid reply (-> HTTP 502)."""


class LlmObject(BaseModel):
    class_name: str
    description: str


class LlmModelRef(BaseModel):
    provider: str
    name: str


class LlmVisionResponse(BaseModel):
    data_origin: Literal["external_llm"] = "external_llm"
    model: LlmModelRef
    used_in_fusion: Literal[False] = False
    flood_visible: bool
    objects: List[LlmObject]
    counts: Dict[str, int]
    summary: str
    latency_ms: float = Field(..., ge=0)


def is_configured() -> bool:
    return bool(config.GROQ_API_KEY)


def _encode(image) -> str:
    """Downscale and JPEG-encode a PIL image as a data URL."""
    img = image.copy()
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parse_reply(content: str) -> Dict[str, Any]:
    """
    Validate the LLM's JSON reply.

    Raises:
        LlmUnavailable: if the reply is not the requested shape or uses an unknown class.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LlmUnavailable(f"LLM reply is not valid JSON: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        raise LlmUnavailable("LLM reply has no 'objects' list")

    objects = []
    for obj in data["objects"]:
        if not isinstance(obj, dict):
            raise LlmUnavailable(f"LLM object is not a JSON object: {obj!r}")
        name = str(obj.get("class_name", "")).strip()
        if name not in CLASS_NAMES:
            raise LlmUnavailable(f"LLM used unknown class '{name}'; expected one of {CLASS_NAMES}")
        objects.append({"class_name": name, "description": str(obj.get("description", "")).strip()})

    counts = {name: 0 for name in CLASS_NAMES}
    for obj in objects:
        counts[obj["class_name"]] += 1
    return {
        "flood_visible": bool(data.get("flood_visible", False)),
        "objects": objects,
        "counts": counts,
        "summary": str(data.get("summary", "")).strip(),
    }


def check_image(image) -> LlmVisionResponse:
    """
    Ask the Groq vision LLM for a second opinion on a PIL image.

    Raises:
        LlmNotConfigured: GROQ_API_KEY missing.
        LlmUnavailable: request failed or reply invalid.
    """
    if not is_configured():
        raise LlmNotConfigured("LLM second opinion not configured: set GROQ_API_KEY in backend/.env")

    payload = {
        "model": config.GROQ_VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": _encode(image)}},
            ],
        }],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    start = time.perf_counter()
    try:
        resp = httpx.post(
            GROQ_URL,
            json=payload,
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            timeout=TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        raise LlmUnavailable(f"Groq request failed: {type(exc).__name__}: {exc}")
    latency_ms = (time.perf_counter() - start) * 1000

    if resp.status_code != 200:
        try:
            reason = resp.json().get("error", {}).get("message", resp.text[:300])
        except ValueError:
            reason = resp.text[:300]
        raise LlmUnavailable(f"Groq returned HTTP {resp.status_code}: {reason}")

    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LlmUnavailable(f"Unexpected Groq response shape: {exc}")

    parsed = parse_reply(content)
    result = LlmVisionResponse(
        model=LlmModelRef(provider=PROVIDER, name=config.GROQ_VISION_MODEL),
        latency_ms=round(latency_ms, 1),
        **parsed,
    )
    logger.info(f"LLM second opinion | model={config.GROQ_VISION_MODEL} counts={result.counts} time={latency_ms:.0f}ms")
    return result
