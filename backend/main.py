"""
MDRIS Backend — Main Application (Flood-Scoped)
==================================================
FastAPI application entry point (docs/MDRIS_CONTEXT.md section 9).

AI endpoints:
  POST /api/vision/predict    — YOLOv8n (COCO-pretrained by default, D-37)
  POST /api/nlp/analyze       — fine-tuned DistilBERT urgency classifier
  POST /api/fusion/analyze    — confidence x reliability weighted fusion
  POST /api/incident/analyze  — vision + NLP + fusion in one call
  GET  /api/weather/          — OpenWeatherMap, context only (never fused)

If a model is missing, its endpoints return HTTP 503 with the reason.
Every successful response carries `data_origin`.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from services.logging_service import configure_logging

# Configure logging before importing any other modules
configure_logging(level=config.LOG_LEVEL)

logger = logging.getLogger("mdris.main")

from ai import registry  # noqa: E402
from data.incident_store import IncidentsUnavailable  # noqa: E402
from routers import (  # noqa: E402
    alerts,
    dashboard,
    evaluation,
    fusion,
    incident,
    incidents,
    map,
    nlp,
    resources,
    system,
    vision,
    weather,
)


# ── Lifespan: load the fine-tuned models once ───────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MDRIS API starting up — loading models...")
    registry.load_all()
    logger.info(
        f"MDRIS API startup complete | vision loaded={registry.VISION.loaded} | nlp loaded={registry.NLP.loaded}"
    )
    yield
    logger.info("MDRIS API shutting down.")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="MDRIS API",
    description="Multimodal Flood Response Intelligence System — decision-support backend",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────

for directory in (config.UPLOAD_FOLDER, config.STATIC_INCIDENTS_DIR, config.VISION_FIGURES_DIR, config.NLP_FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=config.UPLOAD_FOLDER), name="uploads")
app.mount("/static/incidents", StaticFiles(directory=config.STATIC_INCIDENTS_DIR), name="incident-images")
app.mount("/figures/vision", StaticFiles(directory=config.VISION_FIGURES_DIR), name="vision-figures")
app.mount("/figures/nlp", StaticFiles(directory=config.NLP_FIGURES_DIR), name="nlp-figures")

# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(registry.ModelUnavailable)
async def model_unavailable_handler(request: Request, exc: registry.ModelUnavailable):
    logger.error(f"503 on {request.url.path}: {exc}")
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(IncidentsUnavailable)
async def incidents_unavailable_handler(request: Request, exc: IncidentsUnavailable):
    logger.error(f"503 on {request.url.path}: {exc}")
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(incidents.router,  prefix="/api/incidents",     tags=["Incidents (simulated)"])
app.include_router(dashboard.router,  prefix="/api/dashboard",     tags=["Dashboard (simulated)"])
app.include_router(map.router,        prefix="/api/map",           tags=["Map (simulated)"])
app.include_router(alerts.router,     prefix="/api/alerts",        tags=["Alerts (simulated)"])
app.include_router(resources.router,  prefix="/api/resources",     tags=["Resources (static demo)"])
app.include_router(evaluation.router, prefix="/api/evaluation",    tags=["Model Evaluation"])
app.include_router(system.router,     prefix="/api/system-status", tags=["System Status"])

app.include_router(vision.router,   prefix="/api/vision",   tags=["Vision AI"])
app.include_router(nlp.router,      prefix="/api/nlp",      tags=["NLP AI"])
app.include_router(weather.router,  prefix="/api/weather",  tags=["Weather"])
app.include_router(fusion.router,   prefix="/api/fusion",   tags=["Fusion Engine"])
app.include_router(incident.router, prefix="/api/incident", tags=["Incident"])

# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def read_root():
    return {"data_origin": "live", "message": "MDRIS API — flood response decision support", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"data_origin": "live", "status": "ok"}
