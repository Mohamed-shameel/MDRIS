"""
Dashboard Router (simulated)
=============================
GET /api/dashboard — KPIs computed from the simulated incidents in incidents.json.
"""

from fastapi import APIRouter

from ai.fusion.constants import RISK_BANDS
from data import incident_store

router = APIRouter()


@router.get("", summary="KPIs computed from the simulated incidents")
def get_dashboard():
    payload = incident_store.load()
    incidents = payload["incidents"]
    fusions = [i["fusion"] for i in incidents]
    bands = [band for _, band in RISK_BANDS]
    n = len(fusions)
    return {
        "data_origin": "simulated",
        "generated_at": payload["generated_at"],
        "kpis": {
            "total_incidents": n,
            "critical": sum(f["risk_band"] == "Critical" for f in fusions),
            "high": sum(f["risk_band"] == "High" for f in fusions),
            "conflicts": sum(f["conflict"] for f in fusions),
            "mean_evidence_strength": round(sum(f["evidence_strength"] for f in fusions) / n, 3) if n else 0.0,
        },
        "band_counts": {band: sum(f["risk_band"] == band for f in fusions) for band in bands},
    }
