"""
Fusion Module — Fusion Engine
===============================
Confidence- and source-reliability-weighted fusion of the vision and text
evidence (docs/MDRIS_CONTEXT.md section 8).

  S_v   = 1 - prod_i (1 - w_c_i * conf_i)      noisy-OR over detections, conf >= 0.25
  S_n   = sum_k p_k * URGENCY_VALUES[k]         expected urgency
  eff_m = MODALITY_WEIGHT_m * reliability_m * conf_m   (0 if the modality is absent)

  priority_score    = 100 * (eff_v*S_v + eff_n*S_n) / (eff_v + eff_n)
  evidence_strength = eff_v + eff_n
  conflict          = both present and |S_v - S_n| > CONFLICT_THRESHOLD

Weather is deliberately NOT an input (section 8.6); a test enforces this.
The rule is documented and auditable, not learned and not calibrated.
"""

import logging
from typing import Dict, List, Optional, Tuple

from . import constants as C
from .actions import suggest_action
from .schemas import (
    FusionBreakdown,
    FusionResponse,
    FusionWeights,
    TextEvidence,
    VisionEvidence,
)

logger = logging.getLogger("mdris.fusion.engine")


# ---------------------------------------------------------------------------
# Per-modality scores
# ---------------------------------------------------------------------------

def _kept_detections(detections) -> List[Tuple[str, float]]:
    """(class_name, confidence) pairs at or above the detection threshold."""
    kept = []
    for det in detections:
        name = det["class_name"] if isinstance(det, dict) else det.class_name
        conf = float(det["confidence"] if isinstance(det, dict) else det.confidence)
        if name not in C.CLASS_WEIGHTS:
            raise ValueError(f"Unknown vision class '{name}'. Expected one of {list(C.CLASS_WEIGHTS)}")
        if conf >= C.DETECTION_CONF_THRESHOLD:
            kept.append((name, conf))
    return kept


def vision_score(detections) -> Tuple[float, float]:
    """
    Noisy-OR vision score.

    Returns:
        (S_v, conf_v). With no detections: (0.0, NO_DETECTION_CONFIDENCE).
    """
    kept = _kept_detections(detections)
    if not kept:
        return 0.0, C.NO_DETECTION_CONFIDENCE
    miss = 1.0
    for name, conf in kept:
        miss *= 1.0 - C.CLASS_WEIGHTS[name] * conf
    return 1.0 - miss, max(conf for _, conf in kept)


def text_score(probabilities: Dict[str, float]) -> Tuple[float, float]:
    """
    Expected-urgency text score.

    Returns:
        (S_n, conf_n) where conf_n is the top class probability.
    """
    missing = [label for label in C.URGENCY_LABELS if label not in probabilities]
    if missing:
        raise ValueError(f"Urgency probabilities missing labels: {missing}")
    s_n = sum(float(probabilities[k]) * C.URGENCY_VALUES[k] for k in C.URGENCY_LABELS)
    conf_n = max(float(probabilities[k]) for k in C.URGENCY_LABELS)
    return s_n, conf_n


def dominant_class(detections) -> str:
    """argmax over detections of w_c * conf, or "text_only" if there are none."""
    kept = _kept_detections(detections)
    if not kept:
        return "text_only"
    name, _ = max(kept, key=lambda nc: C.CLASS_WEIGHTS[nc[0]] * nc[1])
    return name


def risk_band(priority_score: float) -> str:
    """First band in RISK_BANDS whose threshold the score reaches."""
    for threshold, band in C.RISK_BANDS:
        if priority_score >= threshold:
            return band
    return C.RISK_BANDS[-1][1]


def evidence_level(evidence_strength: float) -> str:
    for threshold, level in C.EVIDENCE_LEVELS:
        if evidence_strength >= threshold:
            return level
    return C.EVIDENCE_LEVELS[-1][1]


def _reliability(table: Dict[str, float], source: str, kind: str) -> float:
    if source not in table:
        raise ValueError(f"Unknown {kind} source '{source}'. Expected one of {list(table)}")
    return table[source]


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------

def fuse(
    vision: Optional[VisionEvidence] = None,
    nlp: Optional[TextEvidence] = None,
    *,
    vision_weight: float = C.VISION_WEIGHT,
    nlp_weight: float = C.NLP_WEIGHT,
    vision_reliability: Optional[Dict[str, float]] = None,
    text_reliability: Optional[Dict[str, float]] = None,
) -> FusionResponse:
    """
    Fuse vision and text evidence into a priority decision.

    The keyword-only weight / reliability overrides exist for the sensitivity
    analysis in training/fusion_analysis.py; the API always uses the defaults.

    Raises:
        ValueError: if both modalities are missing, or a source / class is unknown.
    """
    if vision is None and nlp is None:
        raise ValueError("At least one of 'vision' or 'nlp' must be provided.")

    r_vision = vision_reliability or C.VISION_SOURCE_RELIABILITY
    r_text = text_reliability or C.TEXT_SOURCE_RELIABILITY

    s_v = conf_v = rel_v = None
    s_n = conf_n = rel_n = None
    eff_v = eff_n = 0.0

    if vision is not None:
        s_v, conf_v = vision_score(vision.detections)
        rel_v = _reliability(r_vision, vision.source, "vision")
        eff_v = vision_weight * rel_v * conf_v

    if nlp is not None:
        s_n, conf_n = text_score(nlp.probabilities)
        rel_n = _reliability(r_text, nlp.source, "text")
        eff_n = nlp_weight * rel_n * conf_n

    total = eff_v + eff_n
    if total <= 0.0:
        raise ValueError("Evidence strength is zero; cannot compute a priority score.")

    weighted = (eff_v * s_v if s_v is not None else 0.0) + (eff_n * s_n if s_n is not None else 0.0)
    priority = 100.0 * weighted / total

    both = vision is not None and nlp is not None
    conflict = bool(both and abs(s_v - s_n) > C.CONFLICT_THRESHOLD)
    band = risk_band(priority)
    dominant = dominant_class(vision.detections) if vision is not None else "text_only"

    result = FusionResponse(
        priority_score=round(priority, 1),
        risk_band=band,
        evidence_strength=round(total, 3),
        evidence_level=evidence_level(total),
        conflict=conflict,
        single_source=not both,
        dominant_class=dominant,
        suggested_action=suggest_action(dominant, band, conflict),
        breakdown=FusionBreakdown(
            S_v=None if s_v is None else round(s_v, 4),
            conf_v=None if conf_v is None else round(conf_v, 4),
            reliability_v=rel_v,
            eff_v=round(eff_v, 4),
            S_n=None if s_n is None else round(s_n, 4),
            conf_n=None if conf_n is None else round(conf_n, 4),
            reliability_n=rel_n,
            eff_n=round(eff_n, 4),
            weights=FusionWeights(vision=vision_weight, nlp=nlp_weight),
        ),
    )

    logger.info(
        f"Fusion | score={result.priority_score} band={band} evidence={result.evidence_strength} "
        f"conflict={conflict} dominant={dominant}"
    )
    return result
