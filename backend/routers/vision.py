"""
Vision Router
==============
POST /api/vision/predict   — served YOLOv8n (COCO-pretrained by default, D-37) on a flood image
POST /api/vision/llm-check — Groq LLM second opinion (data_origin external_llm, never fused)
GET  /api/vision/status    — real load state of the checkpoint
"""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from ai import registry
from ai.fusion.constants import DEFAULT_VISION_SOURCE
from ai.llm_vision import service as llm_vision
from ai.vision.image_utils import SUPPORTED_FORMATS, open_image, validate_image_bytes
from ai.vision.schemas import VisionResponse
from ai.vision.service import process_image_upload

logger = logging.getLogger("mdris.router.vision")

router = APIRouter()


async def read_upload(file: UploadFile) -> tuple[bytes, str]:
    """Read an uploaded image, rejecting unsupported MIME types with 422."""
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: JPEG, PNG.",
        )
    try:
        return await file.read(), content_type
    except Exception as exc:
        logger.error(f"Failed to read uploaded file: {exc}")
        raise HTTPException(status_code=400, detail="Could not read uploaded file.")


@router.post("/predict", response_model=VisionResponse, summary="Detect people and vehicles in danger in a flood image")
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPEG or PNG, max 15 MB)"),
    source: str = Form(DEFAULT_VISION_SOURCE, description="officer_photo | drone | cctv | citizen_upload"),
) -> VisionResponse:
    image_data, content_type = await read_upload(file)
    try:
        return process_image_upload(image_data, content_type, source)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post(
    "/llm-check",
    response_model=llm_vision.LlmVisionResponse,
    summary="LLM second opinion (Groq) on a flood image; never used by fusion",
)
async def llm_check(
    file: UploadFile = File(..., description="Image file (JPEG or PNG, max 15 MB)"),
) -> llm_vision.LlmVisionResponse:
    image_data, content_type = await read_upload(file)
    try:
        validate_image_bytes(image_data, content_type)
        image = open_image(image_data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        return await run_in_threadpool(llm_vision.check_image, image)
    except llm_vision.LlmNotConfigured as exc:
        logger.error(f"503 on /api/vision/llm-check: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))
    except llm_vision.LlmUnavailable as exc:
        logger.error(f"502 on /api/vision/llm-check: {exc}")
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/status", summary="Vision model load state")
def vision_status():
    return {"data_origin": "live", **registry.VISION.public_status()}
