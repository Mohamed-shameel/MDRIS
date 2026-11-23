"""
Fusion Router
==============
POST /api/fusion/analyze

Fuses precomputed vision detections and/or urgency probabilities into a
priority score, risk band, evidence strength and conflict flag.
Both inputs missing -> HTTP 422.
"""

import logging

from fastapi import APIRouter, HTTPException

from ai.fusion.fusion_engine import fuse
from ai.fusion.schemas import FusionRequest, FusionResponse

logger = logging.getLogger("mdris.router.fusion")

router = APIRouter()


@router.post("/analyze", response_model=FusionResponse, summary="Fuse vision and text evidence into a priority decision")
def analyze_fusion(request: FusionRequest) -> FusionResponse:
    try:
        return fuse(request.vision, request.nlp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
