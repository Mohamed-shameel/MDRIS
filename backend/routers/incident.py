"""
Incident Router
================
POST /api/incident/analyze

One round trip for the Incident Analyzer page: optional image + optional text
-> YOLOv8n (COCO-pretrained by default, D-37) + fine-tuned DistilBERT + fusion.
If an input is given but its model is not loaded, the request fails with 503;
nothing is silently skipped.
"""

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ai.fusion.constants import DEFAULT_TEXT_SOURCE, DEFAULT_VISION_SOURCE
from ai.fusion.fusion_engine import fuse
from ai.fusion.schemas import TextEvidence, VisionEvidence
from ai.nlp.service import analyze_text
from ai.vision.service import process_image_upload
from routers.vision import read_upload

logger = logging.getLogger("mdris.router.incident")

router = APIRouter()


@router.post("/analyze", summary="Run vision + NLP + fusion on one incident")
async def analyze_incident(
    image: Optional[UploadFile] = File(None, description="Optional image (JPEG or PNG)"),
    text: Optional[str] = Form(None, description="Optional incident text (3-2000 chars)"),
    image_source: str = Form(DEFAULT_VISION_SOURCE),
    text_source: str = Form(DEFAULT_TEXT_SOURCE),
):
    text = (text or "").strip() or None
    has_image = image is not None and bool(image.filename)
    if not has_image and text is None:
        raise HTTPException(status_code=422, detail="Provide an image, a text, or both.")
    if text is not None and not 3 <= len(text) <= 2000:
        raise HTTPException(status_code=422, detail="Text must be 3-2000 characters.")

    vision_result = nlp_result = None
    try:
        if has_image:
            image_data, content_type = await read_upload(image)
            vision_result = process_image_upload(image_data, content_type, image_source)
        if text is not None:
            nlp_result = analyze_text(text, text_source)

        fusion_result = fuse(
            VisionEvidence(
                detections=[{"class_name": d.class_name, "confidence": d.confidence} for d in vision_result.detections],
                source=vision_result.source,
            ) if vision_result else None,
            TextEvidence(probabilities=nlp_result.probabilities, source=nlp_result.source) if nlp_result else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "data_origin": "model",
        "vision": vision_result,
        "nlp": nlp_result,
        "fusion": fusion_result,
    }
