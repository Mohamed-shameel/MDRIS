"""
API smoke test: every endpoint of docs/MDRIS_CONTEXT.md section 9.3.

Asserts status codes and that every JSON body carries `data_origin`.
A 503 from a model endpoint only passes if that model is listed in BLOCKED
(a documented blocker in docs/MORNING_REPORT.md); otherwise it fails.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main

# Models that are documented as BLOCKED (missing data). Empty = both must work.
BLOCKED: set[str] = set()

ORIGINS = {"model", "simulated", "static_demo", "live", "external_llm"}


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:  # runs the lifespan: loads both models
        yield c


def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (90, 110, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def check(response, expected_origin: str, model: str | None = None):
    if response.status_code == 503 and model in BLOCKED:
        pytest.skip(f"{model} model is BLOCKED: {response.json()['detail']}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("data_origin") in ORIGINS
    assert body["data_origin"] == expected_origin, body["data_origin"]
    return body


# ── Live / static ────────────────────────────────────────────────────────────

def test_health(client):
    check(client.get("/health"), "live")


def test_vision_status(client):
    body = check(client.get("/api/vision/status"), "live")
    assert body["loaded"] or "vision" in BLOCKED


def test_nlp_labels(client):
    body = check(client.get("/api/nlp/labels"), "live")
    assert body["labels"] == ["Low", "Medium", "High"]


def test_weather_without_key_or_live(client):
    response = client.get("/api/weather/", params={"lat": 13.0827, "lon": 80.2707})
    assert response.status_code in (200, 502)
    if response.status_code == 200:
        body = check(response, "live")
        assert "configured" in body


def test_weather_presets(client):
    check(client.get("/api/weather/presets"), "live")


def test_system_status(client):
    body = check(client.get("/api/system-status"), "live")
    assert set(body["models"]) == {"vision", "nlp"}


def test_resources_static_demo(client):
    check(client.get("/api/resources"), "static_demo")


def test_analytics_removed(client):
    assert client.get("/api/analytics").status_code == 404


# ── Models ───────────────────────────────────────────────────────────────────

def test_vision_predict(client):
    response = client.post(
        "/api/vision/predict",
        files={"file": ("smoke.jpg", jpeg_bytes(), "image/jpeg")},
        data={"source": "cctv"},
    )
    body = check(response, "model", "vision")
    assert body["source"] == "cctv"
    assert 0.0 <= body["vision_score"] <= 1.0
    assert set(body["counts"]) == {"person_in_water", "person_in_vehicle", "person", "vehicle"}


def test_served_vision_model_is_labelled_truthfully(client):
    """D-37: the COCO-pretrained model must never be reported as fine-tuned, and vice versa."""
    import config
    coco = config.VISION_MODEL_VARIANT == "coco_pretrained"
    status = check(client.get("/api/vision/status"), "live")
    assert status["fine_tuned"] is (not coco)
    assert ("coco" in status["name"]) is coco
    response = client.post("/api/vision/predict", files={"file": ("smoke.jpg", jpeg_bytes(), "image/jpeg")})
    body = check(response, "model", "vision")
    assert body["model"]["fine_tuned"] is (not coco)
    assert body["model"]["name"] == status["name"]


def test_llm_check_without_key_is_503(client, monkeypatch):
    import config
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    response = client.post("/api/vision/llm-check", files={"file": ("smoke.jpg", jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_llm_check_is_labelled_external(client, monkeypatch):
    """Groq mocked: the reply must be tagged external_llm and never claim to be used in fusion."""
    import config
    import httpx

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content":
                '{"flood_visible": true, "objects": [{"class_name": "person_in_water", "description": "wading"}],'
                ' "summary": "one person wading"}'}}]}

    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse())
    response = client.post("/api/vision/llm-check", files={"file": ("smoke.jpg", jpeg_bytes(), "image/jpeg")})
    body = check(response, "external_llm")
    assert body["used_in_fusion"] is False
    assert body["counts"]["person_in_water"] == 1
    assert body["model"]["provider"] == "Groq"


def test_vision_rejects_bad_type(client):
    response = client.post("/api/vision/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert response.status_code == 422


def test_nlp_analyze(client):
    response = client.post("/api/nlp/analyze", json={"text": "Family trapped on the roof, water rising, please send a boat", "source": "citizen_sms"})
    body = check(response, "model", "nlp")
    assert body["urgency"] in ("Low", "Medium", "High")
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_nlp_rejects_short_text(client):
    assert client.post("/api/nlp/analyze", json={"text": "hi"}).status_code == 422


def test_fusion_analyze(client):
    response = client.post("/api/fusion/analyze", json={
        "vision": {"detections": [{"class_name": "person", "confidence": 0.8}], "source": "drone"},
        "nlp": {"probabilities": {"Low": 0.2, "Medium": 0.3, "High": 0.5}, "source": "social_media"},
    })
    body = check(response, "model")
    assert body["risk_band"] in ("Critical", "High", "Moderate", "Low")


def test_fusion_both_missing_is_422(client):
    assert client.post("/api/fusion/analyze", json={}).status_code == 422


def test_incident_analyze_both(client):
    response = client.post(
        "/api/incident/analyze",
        files={"image": ("smoke.jpg", jpeg_bytes(), "image/jpeg")},
        data={"text": "Road flooded near the bridge, two cars stuck", "image_source": "officer_photo",
              "text_source": "control_room_call"},
    )
    if response.status_code == 503 and BLOCKED:
        pytest.skip(response.json()["detail"])
    body = check(response, "model")
    assert body["vision"]["data_origin"] == "model"
    assert body["nlp"]["data_origin"] == "model"
    assert body["fusion"]["data_origin"] == "model"
    assert body["fusion"]["single_source"] is False


def test_incident_analyze_nothing_is_422(client):
    assert client.post("/api/incident/analyze", data={"image_source": "drone"}).status_code == 422


# ── Simulated incidents ─────────────────────────────────────────────────────

def test_incidents_sorted(client):
    body = check(client.get("/api/incidents"), "simulated")
    scores = [(i["fusion"]["priority_score"], i["fusion"]["evidence_strength"]) for i in body["incidents"]]
    assert scores == sorted(scores, reverse=True)
    assert body["count"] == len(body["incidents"]) > 0
    assert all(i["simulated"] is True for i in body["incidents"])


def test_incident_by_id(client):
    first = client.get("/api/incidents").json()["incidents"][0]["id"]
    body = check(client.get(f"/api/incidents/{first}"), "simulated")
    assert body["id"] == first
    assert client.get("/api/incidents/SIM-9999").status_code == 404


def test_dashboard(client):
    body = check(client.get("/api/dashboard"), "simulated")
    assert body["kpis"]["total_incidents"] == sum(body["band_counts"].values())


def test_map(client):
    body = check(client.get("/api/map"), "simulated")
    assert all({"lat", "lon", "risk_band", "evidence_strength"} <= set(f) for f in body["features"])


def test_alerts(client):
    body = check(client.get("/api/alerts"), "simulated")
    assert all(i["fusion"]["risk_band"] in ("High", "Critical") for i in body["incidents"])


def test_evaluation(client):
    body = check(client.get("/api/evaluation"), "model")
    for part in ("vision", "nlp"):
        if part not in BLOCKED:
            assert body[part]["available"], body[part]["reason"]
    assert body["fusion"]["available"], body["fusion"]["reason"]


def test_static_figures_and_incident_images(client):
    body = client.get("/api/evaluation").json()
    for url in list(body["vision"]["figure_urls"].values())[:2] + list(body["nlp"]["figure_urls"].values())[:2]:
        assert client.get(url).status_code == 200, url
    incident = client.get("/api/incidents").json()["incidents"]
    with_image = next((i for i in incident if i["image"]), None)
    if with_image:
        assert client.get(with_image["image"]["url"]).status_code == 200
