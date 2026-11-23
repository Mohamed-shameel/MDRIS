"""
Evaluation Router (model)
==========================
GET /api/evaluation — the saved test metrics of both fine-tuned models
(models/*/metrics.json), their figure URLs, and the fusion ablation and
sensitivity analysis (models/fusion_analysis.json). Replaces /api/analytics.

A missing file is reported as available=false with the reason; no numbers
are filled in.
"""

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

import config

router = APIRouter()


def _read(path: Path) -> tuple[Any, str | None]:
    if not path.exists():
        return None, f"{path.relative_to(config.REPO_ROOT).as_posix()} not found (model not trained/evaluated yet)"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"could not parse {path.name}: {exc}"


def _model_block(metrics_path: Path, figures_dir: Path, url_prefix: str) -> Dict[str, Any]:
    metrics, reason = _read(metrics_path)
    figure_urls = {}
    for rel in (metrics or {}).get("figures", []):
        inner = rel.split("figures/", 1)[-1]
        if (figures_dir / inner).exists():
            figure_urls[rel] = f"{url_prefix}/{inner}"
    return {"available": metrics is not None, "reason": reason, "metrics": metrics, "figure_urls": figure_urls}


@router.get("", summary="Test metrics, baselines, figures and fusion analysis")
def get_evaluation():
    analysis, reason = _read(config.FUSION_ANALYSIS_PATH)
    return {
        "data_origin": "model",
        "vision": _model_block(config.VISION_METRICS_PATH, config.VISION_FIGURES_DIR, "/figures/vision"),
        "nlp": _model_block(config.NLP_METRICS_PATH, config.NLP_FIGURES_DIR, "/figures/nlp"),
        "fusion": {"available": analysis is not None, "reason": reason, "analysis": analysis},
    }
