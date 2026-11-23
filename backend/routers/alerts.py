"""
Alerts Router (simulated)
==========================
GET /api/alerts — simulated incidents in the High or Critical band, newest first.
"""

from fastapi import APIRouter

from data import incident_store

router = APIRouter()

ALERT_BANDS = {"Critical", "High"}


@router.get("", summary="High and Critical simulated incidents, newest first")
def get_alerts():
    payload = incident_store.load()
    alerts = sorted(
        (i for i in payload["incidents"] if i["fusion"]["risk_band"] in ALERT_BANDS),
        key=lambda i: i["created_at"],
        reverse=True,
    )
    return {
        "data_origin": "simulated",
        "generated_at": payload["generated_at"],
        "count": len(alerts),
        "incidents": alerts,
    }
