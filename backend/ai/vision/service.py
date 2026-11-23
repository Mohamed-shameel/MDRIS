"""
Vision Module — Service Layer
================================
Upload pipeline used by the routers:
  1. Validate & open the uploaded image
  2. Run the served YOLOv8n (ai.vision.predict)
  3. Save the annotated image to the uploads folder
  4. Return the section 9.4 vision response
"""

import logging
import uuid

import config
from .image_utils import open_image, save_pil_image, validate_image_bytes
from .predict import predict, validate_source
from .schemas import VisionResponse

logger = logging.getLogger("mdris.vision.service")


def process_image_upload(image_data: bytes, content_type: str, source: str) -> VisionResponse:
    """
    Full pipeline: validate -> open -> infer -> annotate -> save -> return.

    Raises:
        ValueError: bad image or unknown source (-> 422).
        registry.ModelUnavailable: model not loaded (-> 503).
    """
    validate_image_bytes(image_data, content_type)
    validate_source(source)
    img = open_image(image_data)

    result, annotated = predict(img, source)

    config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}_annotated.jpg"
    if save_pil_image(annotated, str(config.UPLOAD_FOLDER / filename)):
        result["annotated_image_url"] = f"/uploads/{filename}"

    logger.info(
        f"Vision inference | detections={len(result['detections'])} S_v={result['vision_score']:.3f} "
        f"band={result['severity_band']} time={result['inference_ms']:.1f}ms"
    )
    return VisionResponse(**result)
