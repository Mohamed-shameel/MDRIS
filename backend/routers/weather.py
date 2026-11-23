"""
Weather Router
===============
GET /api/weather/?lat=&lon=   — current weather (context only, never used by fusion)
GET /api/weather/presets      — preset flood-prone city coordinates
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from ai.weather.service import WeatherNotConfigured, WeatherResponse, WeatherUnavailable, fetch_weather

logger = logging.getLogger("mdris.router.weather")

router = APIRouter()

# Preset locations for quick lookup
PRESET_LOCATIONS = {
    "chennai":   {"lat": 13.0827, "lon": 80.2707, "name": "Chennai"},
    "mumbai":    {"lat": 19.0760, "lon": 72.8777, "name": "Mumbai"},
    "kolkata":   {"lat": 22.5726, "lon": 88.3639, "name": "Kolkata"},
    "delhi":     {"lat": 28.7041, "lon": 77.1025, "name": "New Delhi"},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946, "name": "Bengaluru"},
    "bhubaneswar": {"lat": 20.2961, "lon": 85.8245, "name": "Bhubaneswar"},
    "guwahati":  {"lat": 26.1445, "lon": 91.7362, "name": "Guwahati"},
}


@router.get("/", response_model=WeatherResponse | WeatherNotConfigured, summary="Current weather (context only)")
async def get_weather(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude"),
):
    try:
        return await fetch_weather(lat=lat, lon=lon)
    except WeatherUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/presets", summary="Preset flood-prone city coordinates")
def get_presets():
    return {"data_origin": "live", "presets": PRESET_LOCATIONS}
