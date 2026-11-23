"""
Vision Module — Model Loader
==============================
Loads the YOLOv8n the backend serves (docs/MDRIS_CONTEXT.md D-37):
  - coco_pretrained: backend/yolov8n.pt (80 COCO classes, mapped by COCO_TO_CANONICAL)
  - finetuned:       models/vision/best.pt (the 4 canonical classes)
Raises on any problem; ai/registry.py records the error and the API
returns HTTP 503. There is no fallback model.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

from ai.fusion.constants import CLASS_NAMES, COCO_TO_CANONICAL

logger = logging.getLogger("mdris.vision.model_loader")

# ultralytics 8.2.60 predates torch 2.6, which made torch.load(weights_only=True) the
# default. Our checkpoints are full pickled models (ours or the official yolov8n.pt).
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

COCO_EXPECTED = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_yolo(path: Path, variant: str) -> Tuple[object, Dict[int, str]]:
    """
    Load a YOLO checkpoint and check its classes.

    Returns:
        (model, class_map) where class_map maps the model's class id -> canonical class name.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"checkpoint not found at {path}")

    from ultralytics import YOLO

    model = YOLO(str(path))
    if variant == "finetuned":
        names = [model.names[i] for i in sorted(model.names)]
        if names != CLASS_NAMES:
            raise ValueError(f"checkpoint classes {names} do not match canonical classes {CLASS_NAMES}")
        return model, dict(enumerate(CLASS_NAMES))

    if variant == "coco_pretrained":
        if len(model.names) != 80 or any(model.names.get(i) != n for i, n in COCO_EXPECTED.items()):
            raise ValueError(f"{path} is not the COCO-pretrained yolov8n (unexpected class names)")
        return model, dict(COCO_TO_CANONICAL)

    raise ValueError(f"unknown vision model variant '{variant}'")
