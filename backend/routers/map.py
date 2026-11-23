"""
Map Router (simulated)
=======================
GET /api/map — simulated incidents as map features (GeoJSON-like list).
"""

from fastapi import APIRouter

from data import incident_store

router = APIRouter()


@router.get("", summary="Simulated incidents as map features")
def get_map():
    return {
        "data_origin": "simulated",
        "features": [
            {
                "id": i["id"],
                "lat": i["location"]["lat"],
                "lon": i["location"]["lon"],
                "location": i["location"]["name"],
                "risk_band": i["fusion"]["risk_band"],
                "priority_score": i["fusion"]["priority_score"],
                "evidence_strength": i["fusion"]["evidence_strength"],
                "conflict": i["fusion"]["conflict"],
                "suggested_action": i["fusion"]["suggested_action"],
            }
            for i in incident_store.sorted_incidents()
        ],
    }
