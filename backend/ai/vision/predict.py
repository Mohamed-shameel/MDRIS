"""
Vision Module — Core Inference
================================
Runs the served YOLOv8n (COCO-pretrained or fine-tuned, D-37), maps its
classes to the 4 canonical classes and builds the vision response of
docs/MDRIS_CONTEXT.md section 9.4.

The vision score is computed by ai.fusion.fusion_engine.vision_score, so the
API, the incident generator and the fusion rule all use the same formula.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from ai import registry
from ai.fusion.constants import (
    CLASS_NAMES,
    DEFAULT_VISION_SOURCE,
    DETECTION_CONF_THRESHOLD,
    VISION_SOURCE_RELIABILITY,
)
from ai.fusion.fusion_engine import risk_band, vision_score

logger = logging.getLogger("mdris.vision.predict")

# Box colour per canonical class (matches the risk colours used in the UI)
CLASS_COLORS = {
    "person_in_water":   "#DC2626",
    "person_in_vehicle": "#EA580C",
    "person":            "#CA8A04",
    "vehicle":           "#2563EB",
}


def validate_source(source: str) -> str:
    if source not in VISION_SOURCE_RELIABILITY:
        raise ValueError(f"Unknown image source '{source}'. Expected one of {list(VISION_SOURCE_RELIABILITY)}")
    return source


def detect(image) -> Tuple[List[Dict[str, Any]], float]:
    """
    Run YOLO on a PIL image.

    Returns:
        (detections, inference_ms); each detection has class_name, confidence and a pixel bbox.

    Raises:
        registry.ModelUnavailable: if the vision model is not loaded.
    """
    state = registry.require_vision()
    start = time.perf_counter()
    class_map = state.extra["class_map"]
    result = state.model.predict(
        image,
        conf=DETECTION_CONF_THRESHOLD,
        classes=sorted(class_map),  # COCO: only the classes that map to a canonical class
        device=0 if state.device == "cuda" else "cpu",
        verbose=False,
    )[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in class_map:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        detections.append({
            "class_name": class_map[cls_id],
            "confidence": round(float(box.conf.item()), 4),
            "bbox": {"x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2)},
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections, elapsed_ms


def annotate(image, detections: List[Dict[str, Any]]):
    """Return a copy of the image with boxes and class labels drawn."""
    from PIL import ImageDraw, ImageFont

    out = image.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        color = CLASS_COLORS.get(det["class_name"], "#64748b")
        b = det["bbox"]
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=color, width=3)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        text_y = max(0, b["y1"] - 18)
        text_box = draw.textbbox((b["x1"], text_y), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((b["x1"], text_y), label, fill="white", font=font)
    return out


def predict(image, source: str = DEFAULT_VISION_SOURCE) -> Tuple[Dict[str, Any], Any]:
    """
    Full vision prediction for one PIL image.

    Returns:
        (response_dict, annotated_pil_image). `annotated_image_url` is left None;
        the caller decides where the annotated image is saved.
    """
    validate_source(source)
    detections, elapsed_ms = detect(image)
    s_v, conf_v = vision_score(detections)
    counts = {name: 0 for name in CLASS_NAMES}
    for det in detections:
        counts[det["class_name"]] += 1

    response = {
        "data_origin": "model",
        "model": registry.model_info(registry.VISION),
        "source": source,
        "detections": detections,
        "counts": counts,
        "vision_score": round(s_v, 4),
        "vision_confidence": round(conf_v, 4),
        "severity_band": risk_band(100.0 * s_v),
        "annotated_image_url": None,
        "inference_ms": round(elapsed_ms, 1),
    }
    return response, annotate(image, detections)
