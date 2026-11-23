"""
Fusion Module — Pydantic Schemas
==================================
Request and response models for POST /api/fusion/analyze
(docs/MDRIS_CONTEXT.md section 9.4).
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .constants import DEFAULT_TEXT_SOURCE, DEFAULT_VISION_SOURCE


class DetectionInput(BaseModel):
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class VisionEvidence(BaseModel):
    """Vision model output as seen by fusion: detections plus the image source."""
    detections: List[DetectionInput] = Field(default_factory=list)
    source: str = DEFAULT_VISION_SOURCE


class TextEvidence(BaseModel):
    """NLP model output as seen by fusion: urgency probabilities plus the text source."""
    probabilities: Dict[str, float]
    source: str = DEFAULT_TEXT_SOURCE


class FusionRequest(BaseModel):
    vision: Optional[VisionEvidence] = None
    nlp: Optional[TextEvidence] = None


class FusionWeights(BaseModel):
    vision: float
    nlp: float


class FusionBreakdown(BaseModel):
    S_v: Optional[float]
    conf_v: Optional[float]
    reliability_v: Optional[float]
    eff_v: float
    S_n: Optional[float]
    conf_n: Optional[float]
    reliability_n: Optional[float]
    eff_n: float
    weights: FusionWeights


class FusionResponse(BaseModel):
    data_origin: Literal["model"] = "model"
    priority_score: float
    risk_band: str
    evidence_strength: float
    evidence_level: str
    conflict: bool
    single_source: bool
    dominant_class: str
    suggested_action: str
    breakdown: FusionBreakdown
