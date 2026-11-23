"""
Model Registry
================
Loads the served YOLOv8n (COCO-pretrained or fine-tuned, D-37) and the
fine-tuned DistilBERT once at startup and exposes their real load state,
checkpoint path and saved metrics (docs/MDRIS_CONTEXT.md section 9.2).

If a model fails to load it is logged at ERROR and marked loaded=False.
Endpoints that need it raise ModelUnavailable, which the API turns into
HTTP 503. There is no fallback result.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import config

logger = logging.getLogger("mdris.registry")


class ModelUnavailable(RuntimeError):
    """Raised when an endpoint needs a model that is not loaded (-> HTTP 503)."""


@dataclass
class ModelState:
    kind: str                      # "vision" | "nlp"
    name: str
    path: Path
    metrics_path: Path
    loaded: bool = False
    error: Optional[str] = "not loaded yet"
    device: Optional[str] = None
    model: Any = None
    tokenizer: Any = None
    metrics: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def trained_at(self) -> Optional[str]:
        if not self.fine_tuned:
            return None  # official Ultralytics COCO weights; not trained by us
        return (self.metrics or {}).get("trained_at")

    def rel_path(self) -> str:
        try:
            return self.path.relative_to(config.REPO_ROOT).as_posix()
        except ValueError:
            return str(self.path)

    @property
    def fine_tuned(self) -> bool:
        return not (self.kind == "vision" and config.VISION_MODEL_VARIANT == "coco_pretrained")

    def public_status(self) -> Dict[str, Any]:
        headline = None
        test = ((self.metrics or {}).get("metrics") or {}).get("test") or {}
        coco = (((self.metrics or {}).get("baselines") or {}).get("coco_yolov8n") or {}).get("per_class", {})
        if self.kind == "vision" and not self.fine_tuned:
            if "person" in coco:
                headline = {"name": "test person AP50 (COCO baseline)", "value": coco["person"]["AP50"]}
        elif self.kind == "vision" and "mAP50" in test:
            headline = {"name": "test mAP50", "value": test["mAP50"]}
        elif self.kind == "nlp" and "macro_f1" in test:
            headline = {"name": "test macro-F1", "value": test["macro_f1"]}
        return {
            "name": self.name,
            "loaded": self.loaded,
            "error": None if self.loaded else self.error,
            "checkpoint": self.rel_path(),
            "device": self.device,
            "trained_at": self.trained_at,
            "fine_tuned": self.fine_tuned,
            "headline_metric": headline,
        }


def _read_metrics(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"Could not parse metrics file {path}: {exc}")
        return None


VISION = ModelState(
    kind="vision",
    name="yolov8n-coco-pretrained" if config.VISION_MODEL_VARIANT == "coco_pretrained" else "mdris-yolov8n-flood",
    path=config.VISION_MODEL_PATH,
    metrics_path=config.VISION_METRICS_PATH,
)
NLP = ModelState(
    kind="nlp",
    name="mdris-distilbert-urgency",
    path=config.NLP_MODEL_DIR,
    metrics_path=config.NLP_METRICS_PATH,
)

_lock = threading.Lock()


def load_vision() -> ModelState:
    from ai.vision.model_loader import load_yolo

    with _lock:
        VISION.metrics = _read_metrics(VISION.metrics_path)
        try:
            device = config.resolve_device()
            VISION.model, VISION.extra["class_map"] = load_yolo(VISION.path, config.VISION_MODEL_VARIANT)
            VISION.device = device
            VISION.loaded, VISION.error = True, None
            logger.info(f"Vision model loaded from {VISION.path} (variant={config.VISION_MODEL_VARIANT}, device={device})")
        except Exception as exc:
            VISION.model, VISION.loaded = None, False
            VISION.error = f"{type(exc).__name__}: {exc}"
            logger.error(f"Vision model failed to load from {VISION.path}: {VISION.error}")
    return VISION


def load_nlp() -> ModelState:
    from ai.nlp.predict import load_urgency_model

    with _lock:
        NLP.metrics = _read_metrics(NLP.metrics_path)
        try:
            device = config.resolve_device()
            NLP.model, NLP.tokenizer = load_urgency_model(NLP.path, device)
            NLP.device = device
            NLP.loaded, NLP.error = True, None
            logger.info(f"NLP model loaded from {NLP.path} (device={device})")
        except Exception as exc:
            NLP.model, NLP.tokenizer, NLP.loaded = None, None, False
            NLP.error = f"{type(exc).__name__}: {exc}"
            logger.error(f"NLP model failed to load from {NLP.path}: {NLP.error}")
    return NLP


def load_all() -> None:
    load_vision()
    load_nlp()


def require_vision() -> ModelState:
    if not VISION.loaded:
        raise ModelUnavailable(f"vision model not loaded: {VISION.error}")
    return VISION


def require_nlp() -> ModelState:
    if not NLP.loaded:
        raise ModelUnavailable(f"nlp model not loaded: {NLP.error}")
    return NLP


def model_info(state: ModelState) -> Dict[str, Any]:
    """The `model` block embedded in vision / NLP responses (section 9.4)."""
    info = {"name": state.name, "trained_at": state.trained_at, "fine_tuned": state.fine_tuned}
    info["checkpoint" if state.kind == "vision" else "path"] = state.rel_path()
    return info
