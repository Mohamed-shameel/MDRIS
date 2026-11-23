"""
Fusion rule tests T1-T8 (docs/MDRIS_CONTEXT.md section 8.4).

Expected values are copied from the specification. Tolerance: +-0.5 on scores,
+-0.01 on evidence and effective weights. If a test fails, fix the code, never
the expected values or the constants.
"""

import inspect

import pytest
from fastapi.testclient import TestClient

from ai.fusion import constants as C
from ai.fusion.fusion_engine import fuse, vision_score
from ai.fusion.schemas import TextEvidence, VisionEvidence

HI_TEXT = {"Low": 0.05, "Medium": 0.10, "High": 0.85}
LO_TEXT = {"Low": 0.85, "Medium": 0.10, "High": 0.05}

SCORE_TOL = 0.5
EVIDENCE_TOL = 0.01


def vis(detections, source="officer_photo"):
    return VisionEvidence(
        detections=[{"class_name": c, "confidence": p} for c, p in detections],
        source=source,
    )


def txt(probabilities, source="officer_report"):
    return TextEvidence(probabilities=probabilities, source=source)


def test_t1_both_strong_officer_sources():
    r = fuse(vis([("person_in_water", 0.90)]), txt(HI_TEXT))
    assert r.priority_score == pytest.approx(87.0, abs=SCORE_TOL)
    assert r.risk_band == "Critical"
    assert r.evidence_strength == pytest.approx(0.88, abs=EVIDENCE_TOL)
    assert r.conflict is False
    assert r.single_source is False
    assert r.dominant_class == "person_in_water"
    assert r.suggested_action == "Dispatch rescue boat and NDRF swift-water team immediately"


def test_t2_vision_only_is_renormalized():
    r = fuse(vis([("person_in_water", 0.90)]), None)
    assert r.priority_score == pytest.approx(90.0, abs=SCORE_TOL)
    assert r.risk_band == "Critical"
    assert r.evidence_strength == pytest.approx(0.54, abs=EVIDENCE_TOL)
    assert r.single_source is True
    assert r.conflict is False


def test_t3_text_only():
    r = fuse(None, txt(HI_TEXT))
    assert r.priority_score == pytest.approx(82.2, abs=SCORE_TOL)
    assert r.risk_band == "Critical"
    assert r.evidence_strength == pytest.approx(0.34, abs=EVIDENCE_TOL)
    assert r.dominant_class == "text_only"
    assert r.single_source is True


def test_t4_conflicting_signals():
    r = fuse(vis([("person_in_water", 0.90)]), txt(LO_TEXT))
    assert r.priority_score == pytest.approx(63.8, abs=SCORE_TOL)
    assert r.risk_band == "High"
    assert r.evidence_strength == pytest.approx(0.88, abs=EVIDENCE_TOL)
    assert r.conflict is True
    assert r.suggested_action.endswith("Signals disagree: verify on ground before committing resources.")


def test_t5_reliability_changes_influence_not_blind_penalty():
    a = fuse(vis([("person_in_vehicle", 0.80)], "officer_photo"), txt(LO_TEXT, "citizen_sms"))
    b = fuse(vis([("person_in_vehicle", 0.80)], "citizen_upload"), txt(LO_TEXT, "citizen_sms"))

    assert a.priority_score == pytest.approx(54.4, abs=SCORE_TOL)
    assert a.risk_band == "Moderate"
    assert a.breakdown.eff_v == pytest.approx(0.48, abs=EVIDENCE_TOL)
    assert a.evidence_strength == pytest.approx(0.684, abs=EVIDENCE_TOL)

    assert b.priority_score == pytest.approx(49.0, abs=SCORE_TOL)
    assert b.risk_band == "Moderate"
    assert b.breakdown.eff_v == pytest.approx(0.288, abs=EVIDENCE_TOL)
    assert b.evidence_strength == pytest.approx(0.492, abs=EVIDENCE_TOL)

    assert b.breakdown.eff_v < a.breakdown.eff_v
    assert b.evidence_strength < a.evidence_strength


def test_t6_image_without_detections_and_urgent_text():
    r = fuse(vis([]), txt(HI_TEXT))
    assert r.priority_score == pytest.approx(43.7, abs=SCORE_TOL)
    assert r.risk_band == "Moderate"
    assert r.conflict is True
    assert r.breakdown.S_v == 0.0
    assert r.breakdown.conf_v == pytest.approx(C.NO_DETECTION_CONFIDENCE)


def test_t7_many_vehicles_do_not_reach_high():
    detections = [("vehicle", 0.80)] * 5
    s_v, _ = vision_score([{"class_name": c, "confidence": p} for c, p in detections])
    assert s_v == pytest.approx(0.472, abs=0.005)
    r = fuse(vis(detections), None)
    assert r.priority_score == pytest.approx(47.2, abs=SCORE_TOL)
    assert r.risk_band == "Moderate"
    assert r.risk_band not in ("High", "Critical")


def test_t8_nothing_provided_raises():
    with pytest.raises(ValueError):
        fuse(None, None)


def test_t8_nothing_provided_is_http_422():
    import main

    client = TestClient(main.app)  # no lifespan: models are not needed for fusion
    response = client.post("/api/fusion/analyze", json={"vision": None, "nlp": None})
    assert response.status_code == 422


def test_fusion_has_no_weather_input():
    params = [p.lower() for p in inspect.signature(fuse).parameters]
    assert not any("weather" in p or "rain" in p for p in params), params
    from ai.fusion.schemas import FusionRequest

    assert not any("weather" in f.lower() for f in FusionRequest.model_fields)


def test_detections_below_threshold_are_ignored():
    s_v, conf_v = vision_score([{"class_name": "person_in_water", "confidence": 0.2}])
    assert s_v == 0.0
    assert conf_v == C.NO_DETECTION_CONFIDENCE


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        fuse(vis([("person", 0.9)], source="rumour_mill"), None)
