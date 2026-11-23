"""
NLP Router
===========
POST /api/nlp/analyze   — fine-tuned DistilBERT urgency (Low / Medium / High)
GET  /api/nlp/labels    — urgency labels and the HumAID -> urgency mapping
"""

import logging

from fastapi import APIRouter, HTTPException

from ai.fusion.constants import HUMAID_TO_URGENCY, URGENCY_LABELS, URGENCY_VALUES
from ai.nlp.schemas import NLPRequest, NLPResponse
from ai.nlp.service import analyze_text

logger = logging.getLogger("mdris.router.nlp")

router = APIRouter()


@router.post("/analyze", response_model=NLPResponse, summary="Classify incident text urgency")
def analyze_nlp(request: NLPRequest) -> NLPResponse:
    try:
        return analyze_text(request.text, request.source)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/labels", summary="Urgency labels and the HumAID category mapping")
def get_labels():
    return {
        "data_origin": "live",
        "labels": URGENCY_LABELS,
        "urgency_values": URGENCY_VALUES,
        "humaid_mapping": [
            {"humaid_label": humaid, "urgency": urgency} for humaid, urgency in HUMAID_TO_URGENCY.items()
        ],
    }
