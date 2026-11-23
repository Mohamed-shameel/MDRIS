"""
MDRIS Backend — Configuration
===============================
Reads backend/.env once and resolves every path relative to the repo root
(docs/MDRIS_CONTEXT.md section 9.1).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")


def _repo_path(key: str, default: str) -> Path:
    """Read a path from the environment; relative paths are relative to REPO_ROOT."""
    raw = os.getenv(key, "").strip() or default
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


# Which YOLOv8n the backend serves (D-37): coco_pretrained (default) | finetuned
VISION_MODEL_VARIANT = os.getenv("VISION_MODEL_VARIANT", "").strip().lower() or "coco_pretrained"
if VISION_MODEL_VARIANT not in ("coco_pretrained", "finetuned"):
    raise ValueError(f"VISION_MODEL_VARIANT must be coco_pretrained or finetuned, got '{VISION_MODEL_VARIANT}'")
FINETUNED_VISION_MODEL_PATH = _repo_path("VISION_MODEL_PATH", "models/vision/best.pt")
COCO_VISION_MODEL_PATH = _repo_path("COCO_VISION_MODEL_PATH", "backend/yolov8n.pt")
VISION_MODEL_PATH = COCO_VISION_MODEL_PATH if VISION_MODEL_VARIANT == "coco_pretrained" else FINETUNED_VISION_MODEL_PATH
NLP_MODEL_DIR = _repo_path("NLP_MODEL_DIR", "models/nlp")
INCIDENTS_PATH = _repo_path("INCIDENTS_PATH", "backend/data/incidents.json")
UPLOAD_FOLDER = _repo_path("UPLOAD_FOLDER", "backend/uploads")

DEVICE = os.getenv("DEVICE", "auto").strip().lower() or "auto"
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "").strip()
# LLM vision second opinion (D-36); never used by fusion
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "").strip() or "qwen/qwen3.8-27b"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Fixed locations derived from the layout in section 6
MODELS_DIR = REPO_ROOT / "models"
VISION_FIGURES_DIR = MODELS_DIR / "vision" / "figures"
NLP_FIGURES_DIR = MODELS_DIR / "nlp" / "figures"
VISION_METRICS_PATH = MODELS_DIR / "vision" / "metrics.json"
NLP_METRICS_PATH = MODELS_DIR / "nlp" / "metrics.json"
FUSION_ANALYSIS_PATH = MODELS_DIR / "fusion_analysis.json"
STATIC_INCIDENTS_DIR = BACKEND_DIR / "static" / "incidents"


def resolve_device() -> str:
    """Return "cuda" or "cpu" according to DEVICE (auto | cuda | cpu)."""
    if DEVICE == "cpu":
        return "cpu"
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False
    if DEVICE == "cuda" and not cuda_ok:
        raise RuntimeError("DEVICE=cuda but torch.cuda.is_available() is False")
    return "cuda" if cuda_ok else "cpu"
