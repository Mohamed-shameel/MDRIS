"""
NLP Module — Pydantic Schemas
================================
Request and response models for POST /api/nlp/analyze (docs/MDRIS_CONTEXT.md section 9.4).
"""

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

from ai.fusion.constants import DEFAULT_TEXT_SOURCE


class NLPRequest(BaseModel):
    """Incident text to classify for urgency."""
    text: str = Field(..., min_length=3, max_length=2000, description="Text to analyze")
    source: str = Field(DEFAULT_TEXT_SOURCE, description="officer_report | control_room_call | citizen_sms | social_media")


class NLPModelRef(BaseModel):
    name: str
    path: str
    trained_at: Optional[str] = None


class NLPResponse(BaseModel):
    data_origin: Literal["model"] = "model"
    model: NLPModelRef
    source: str
    urgency: str
    probabilities: Dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)
    nlp_score: float = Field(..., ge=0.0, le=1.0)
    inference_ms: float
