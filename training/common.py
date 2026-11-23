"""
MDRIS training — shared constants and helpers
===============================================
Single source of truth for paths, the seed and the label maps used by every
training script (docs/MDRIS_CONTEXT.md sections 5-7).

The class list, class weights and the HumAID -> urgency map are imported from
backend/ai/fusion/constants.py, so training and the backend cannot drift apart.
"""

import os
import random
import sys
from pathlib import Path

# transformers must not import TensorFlow (installed in the base Python, unused here)
os.environ.setdefault("USE_TF", "0")

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ai.fusion.constants import (  # noqa: E402
    CLASS_NAMES,
    CLASS_WEIGHTS,
    COCO_TO_CANONICAL,
    HUMAID_TO_URGENCY,
    URGENCY_LABELS,
)

SEED = 42

# ── Paths ───────────────────────────────────────────────────────────────────

DATASETS_DIR = REPO_ROOT / "datasets"
RAW_VISION_DIR = DATASETS_DIR / "raw" / "flood_yolo"
RAW_TEXT_DIR = DATASETS_DIR / "raw" / "humaid"
PROCESSED_VISION_DIR = DATASETS_DIR / "processed" / "flood_yolo"
PROCESSED_TEXT_DIR = DATASETS_DIR / "processed" / "flood_text"
SPOT_CHECK_DIR = DATASETS_DIR / "processed" / "spot_check"

RUNS_DIR = REPO_ROOT / "training" / "runs"
MODELS_DIR = REPO_ROOT / "models"
VISION_MODEL_DIR = MODELS_DIR / "vision"
NLP_MODEL_DIR = MODELS_DIR / "nlp"
BASE_YOLO_WEIGHTS = BACKEND_DIR / "yolov8n.pt"

# ── Vision: canonical class remap (section 5.1.1) ───────────────────────────
# Raw names are normalized (lowercase, trimmed, spaces/underscores -> hyphens) first.

RAW_TO_CANONICAL = {
    # GNOSEF "detecting humans in floods"
    "drowning": "person_in_water",
    "in-car":   "person_in_vehicle",
    "person":   "person",
    "car":      "vehicle",
    # t5 "final" (fallback dataset)
    "stuck-in-water": "person_in_water",
    "stuck-in-car":   "person_in_vehicle",
    "not-drowning":   "person",
}
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# ── NLP ─────────────────────────────────────────────────────────────────────

URGENCY_ID = {label: i for i, label in enumerate(URGENCY_LABELS)}
ID_TO_URGENCY = {i: label for i, label in enumerate(URGENCY_LABELS)}

VISION_DATASET = {
    "name": "detecting humans in floods (GNOSEF)",
    "url": "https://universe.roboflow.com/gnosef/detecting-humans-in-floods-gukif",
}
TEXT_DATASET = {
    "name": "HumAID flood events (Alam et al., ICWSM 2021)",
    "url": "https://crisisnlp.qcri.org/humaid_dataset",
    "licence": "CC BY-NC 4.0",
}


def normalize_class_name(raw: str) -> str:
    return raw.strip().lower().replace(" ", "-").replace("_", "-")


def set_seed(seed: int = SEED) -> None:
    """Seed random, numpy and torch (if installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def require_cuda() -> str:
    """Exit with an error unless CUDA is available (CLAUDE.md rule 2: no silent CPU training)."""
    import torch

    if not torch.cuda.is_available():
        sys.exit("ERROR: torch.cuda.is_available() is False. Install CUDA PyTorch (section 7.1); refusing to train on CPU.")
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name} {total_gb:.0f}GB"


def allow_trusted_checkpoints() -> None:
    """
    ultralytics 8.2.60 predates torch 2.6's torch.load(weights_only=True) default.
    Must be called before ultralytics loads a checkpoint; only our own / the COCO
    yolov8n.pt checkpoints are ever loaded.
    """
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def now_iso() -> str:
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")
