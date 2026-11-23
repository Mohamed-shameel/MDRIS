"""
Vision Module — Pydantic Schemas
==================================
Response model for POST /api/vision/predict (docs/MDRIS_CONTEXT.md section 9.4).
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PixelBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: PixelBox


class ModelRef(BaseModel):
    name: str
    checkpoint: str
    trained_at: Optional[str] = None
    fine_tuned: bool


class VisionResponse(BaseModel):
    data_origin: Literal["model"] = "model"
    model: ModelRef
    source: str
    detections: List[Detection]
    counts: Dict[str, int]
    vision_score: float = Field(..., ge=0.0, le=1.0)
    vision_confidence: float = Field(..., ge=0.0, le=1.0)
    severity_band: str
    annotated_image_url: Optional[str] = None
    inference_ms: float
