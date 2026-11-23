"""
Incidents Router (simulated)
=============================
GET /api/incidents        — all simulated incidents, priority desc then evidence desc
GET /api/incidents/{id}   — one incident
"""

from fastapi import APIRouter, HTTPException

from data import incident_store

router = APIRouter()


@router.get("", summary="All simulated incidents (sorted by priority)")
def list_incidents():
    payload = incident_store.load()
    incidents = incident_store.sorted_incidents()
    return {
        "data_origin": "simulated",
        "generated_at": payload["generated_at"],
        "count": len(incidents),
        "incidents": incidents,
    }


@router.get("/{incident_id}", summary="One simulated incident")
def get_incident(incident_id: str):
    for inc in incident_store.load()["incidents"]:
        if inc["id"] == incident_id:
            return {"data_origin": "simulated", **inc}
    raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")
