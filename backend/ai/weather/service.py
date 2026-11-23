"""
Weather Module — Service Layer
================================
Fetches current weather from the OpenWeatherMap API for display.

Weather is CONTEXT ONLY: it is never an input to the fusion score
(docs/MDRIS_CONTEXT.md D-14). Without an API key the service reports
{"configured": false}; on a failed request it raises, and the router
returns HTTP 502. No placeholder values are ever returned.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

import config

logger = logging.getLogger("mdris.weather.service")


class WeatherNotConfigured(BaseModel):
    data_origin: Literal["live"] = "live"
    configured: Literal[False] = False
    detail: str = "Weather API key not configured (set OPENWEATHER_KEY in backend/.env)."


class WeatherResponse(BaseModel):
    """Current weather + a rule-based rainfall/wind indicator, for display only."""

    data_origin: Literal["live"] = "live"
    configured: Literal[True] = True

    # Location
    location_name: str
    latitude: float
    longitude: float

    # Raw metrics
    temperature_c: float = Field(..., description="Temperature in °C")
    feels_like_c: float
    humidity_pct: float = Field(..., ge=0, le=100)
    wind_speed_kmh: float
    rainfall_mm: float = Field(..., description="Rain in last 1h (mm)")
    weather_description: str

    # Derived display indicator (not used by fusion)
    risk_level: str = Field(..., description="None / Low / Moderate / High / Critical")
    risk_score: float = Field(..., ge=0.0, le=1.0)


class WeatherUnavailable(RuntimeError):
    """The weather API request failed (-> HTTP 502)."""


# ---------------------------------------------------------------------------
# Display indicator
# ---------------------------------------------------------------------------

def _compute_risk(rainfall_mm: float, wind_kmh: float, humidity: float) -> tuple[str, float]:
    """
    Simple rule-based weather indicator for the Weather page.
    Returns (label, normalized_score 0–1).
    """
    score = 0.0

    # Rainfall contribution (0–0.5)
    if rainfall_mm >= 50:
        score += 0.5
    elif rainfall_mm >= 20:
        score += 0.35
    elif rainfall_mm >= 10:
        score += 0.20
    elif rainfall_mm >= 2:
        score += 0.10

    # Wind contribution (0–0.3)
    if wind_kmh >= 100:
        score += 0.30
    elif wind_kmh >= 60:
        score += 0.20
    elif wind_kmh >= 30:
        score += 0.10

    # Humidity contribution (0–0.2)
    if humidity >= 90:
        score += 0.20
    elif humidity >= 75:
        score += 0.10

    score = min(score, 1.0)

    if score >= 0.75:
        label = "Critical"
    elif score >= 0.50:
        label = "High"
    elif score >= 0.25:
        label = "Moderate"
    elif score > 0.0:
        label = "Low"
    else:
        label = "None"

    return label, round(score, 3)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

async def fetch_weather(lat: float, lon: float) -> WeatherResponse | WeatherNotConfigured:
    """
    Fetch weather data from OpenWeatherMap.

    Raises:
        WeatherUnavailable: if the API request fails.
    """
    if not config.OPENWEATHER_KEY:
        logger.info("OPENWEATHER_KEY not set; weather is not configured.")
        return WeatherNotConfigured()

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": config.OPENWEATHER_KEY, "units": "metric"}

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.error(f"Weather API request failed: {type(exc).__name__}")
        raise WeatherUnavailable(f"Weather API request failed: {type(exc).__name__}") from exc

    main = data.get("main", {})
    wind = data.get("wind", {})
    rain = data.get("rain", {})
    weather_desc = data.get("weather", [{}])[0].get("description", "")
    city_name = data.get("name", f"{lat:.2f},{lon:.2f}")

    humidity = main.get("humidity", 0.0)
    wind_kmh = round(wind.get("speed", 0.0) * 3.6, 1)
    rainfall_mm = rain.get("1h", 0.0)
    risk_level, risk_score = _compute_risk(rainfall_mm, wind_kmh, humidity)

    return WeatherResponse(
        location_name=city_name,
        latitude=lat,
        longitude=lon,
        temperature_c=round(main.get("temp", 0.0), 1),
        feels_like_c=round(main.get("feels_like", 0.0), 1),
        humidity_pct=float(humidity),
        wind_speed_kmh=wind_kmh,
        rainfall_mm=round(rainfall_mm, 2),
        weather_description=weather_desc.capitalize(),
        risk_level=risk_level,
        risk_score=risk_score,
    )
