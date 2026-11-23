"""
Resources Router (static demo)
===============================
GET /api/resources — fixed demo inventory, not connected to any resource system.
"""

from fastapi import APIRouter

from data.static_resources import NOTE, RESOURCES

router = APIRouter()


@router.get("", summary="Static demo resource inventory")
def get_resources():
    return {"data_origin": "static_demo", "note": NOTE, "resources": RESOURCES}
