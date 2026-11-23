"""
Fusion ablation and sensitivity over the 40 demo incidents (docs/MDRIS_CONTEXT.md section 15.3)
================================================================================================
There is no ground-truth priority, so the rule is evaluated by behaviour and stability.
Every variant re-scores the same stored model outputs (detections, probabilities, sources)
of backend/data/incidents.json and is compared with the full rule by
  - the number of incidents whose risk band changes, and
  - Kendall's tau between the priority scores (all 40, and the default top-10).

Ablation (one change each):
  (a) flat mean     : priority = 100 * mean of the S values present (no confidence, reliability or modality weights)
  (b) no reliability: every source reliability = 1.0
  (c) no renormalization: a missing modality counts as present with S = 0 and eff = its full modality
      weight (the old "absent module lowers the score" behaviour)
  (d) weather at 30%: priority = 0.7 * full priority + 30 * W, with W the old rule-based weather score
      for one constant city-wide reading (42 mm/h rain, 55 km/h wind, 90% humidity -> W = 0.65)

Sensitivity:
  - VISION_WEIGHT in {0.4, 0.5, 0.6, 0.7, 0.8}, NLP_WEIGHT = 1 - VISION_WEIGHT
  - all source reliabilities x 0.8 and x 1.2, capped at 1.0

Writes models/fusion_analysis.json.

Usage:  python training/fusion_analysis.py
"""

import json
import math

from scipy.stats import kendalltau

from common import MODELS_DIR, now_iso

WEATHER_READING = {"rainfall_mm": 42.0, "wind_kmh": 55.0, "humidity": 90.0}
VISION_WEIGHTS = [0.4, 0.5, 0.6, 0.7, 0.8]
RELIABILITY_SCALES = [0.8, 1.2]
TOP_K = 10


def tau(a: list[float], b: list[float]) -> float | None:
    """Kendall tau-b; None when undefined (a constant ranking)."""
    value = float(kendalltau(a, b).statistic)
    return None if math.isnan(value) else round(value, 4)


def main() -> None:
    import config
    from ai.fusion import constants as C
    from ai.fusion.fusion_engine import fuse, risk_band
    from ai.fusion.schemas import TextEvidence, VisionEvidence
    from ai.weather.service import _compute_risk

    if not config.INCIDENTS_PATH.exists():
        raise SystemExit(f"STOP: {config.INCIDENTS_PATH} not found. Run training/generate_incidents.py first.")
    data = json.loads(config.INCIDENTS_PATH.read_text(encoding="utf-8"))
    incidents = data["incidents"]

    def evidence(inc):
        v = inc["vision"]
        n = inc["nlp"]
        ve = VisionEvidence(detections=[{"class_name": d["class_name"], "confidence": d["confidence"]}
                                        for d in v["detections"]], source=v["source"]) if v else None
        te = TextEvidence(probabilities=n["probabilities"], source=n["source"]) if n else None
        return ve, te

    ev = [evidence(inc) for inc in incidents]
    full = [fuse(v, t) for v, t in ev]
    full_scores = [r.priority_score for r in full]
    full_bands = [r.risk_band for r in full]
    top_idx = sorted(range(len(full)), key=lambda i: (-full_scores[i], -full[i].evidence_strength))[:TOP_K]

    def compare(name: str, description: str, scores: list[float]) -> dict:
        bands = [risk_band(s) for s in scores]
        tau_all = tau(full_scores, scores)
        tau_top = tau([full_scores[i] for i in top_idx], [scores[i] for i in top_idx])
        return {
            "variant": name,
            "description": description,
            "band_changes": sum(a != b for a, b in zip(full_bands, bands)),
            "kendall_tau": tau_all,
            "kendall_tau_top10": tau_top,
            "mean_score_shift": round(sum(s - f for s, f in zip(scores, full_scores)) / len(scores), 2),
        }

    # ── Ablation ─────────────────────────────────────────────────────────────
    flat = []
    for r in full:
        present = [s for s in (r.breakdown.S_v, r.breakdown.S_n) if s is not None]
        flat.append(100.0 * sum(present) / len(present))

    ones_v = {k: 1.0 for k in C.VISION_SOURCE_RELIABILITY}
    ones_t = {k: 1.0 for k in C.TEXT_SOURCE_RELIABILITY}
    no_rel = [fuse(v, t, vision_reliability=ones_v, text_reliability=ones_t).priority_score for v, t in ev]

    no_renorm = []
    for r in full:
        b = r.breakdown
        eff_v = b.eff_v if b.S_v is not None else C.VISION_WEIGHT
        eff_n = b.eff_n if b.S_n is not None else C.NLP_WEIGHT
        num = (b.eff_v * b.S_v if b.S_v is not None else 0.0) + (b.eff_n * b.S_n if b.S_n is not None else 0.0)
        no_renorm.append(100.0 * num / (eff_v + eff_n))

    _, w = _compute_risk(WEATHER_READING["rainfall_mm"], WEATHER_READING["wind_kmh"], WEATHER_READING["humidity"])
    weather = [0.7 * s + 30.0 * w for s in full_scores]

    ablation = [
        compare("(a) flat mean", "100 x mean of S_v and S_n present; no confidence, reliability or modality weights", flat),
        compare("(b) no reliability", "all source reliabilities set to 1.0", no_rel),
        compare("(c) no renormalization", "missing modality counted as S = 0 with its full modality weight", no_renorm),
        compare("(d) weather at 30%", f"0.7 x priority + 30 x W, constant city reading {WEATHER_READING} -> W = {w}", weather),
    ]

    # ── Sensitivity ──────────────────────────────────────────────────────────
    def sens_row(setting: str, scores: list[float]) -> dict:
        row = compare(setting, "", scores)
        return {"setting": setting, "kendall_tau_top10": row["kendall_tau_top10"],
                "kendall_tau_all": row["kendall_tau"], "band_changes": row["band_changes"]}

    by_weight = [sens_row(f"VISION_WEIGHT={vw:.1f}, NLP_WEIGHT={1 - vw:.1f}",
                          [fuse(v, t, vision_weight=vw, nlp_weight=round(1 - vw, 2)).priority_score for v, t in ev])
                 for vw in VISION_WEIGHTS]
    by_reliability = []
    for scale in RELIABILITY_SCALES:
        rv = {k: min(1.0, x * scale) for k, x in C.VISION_SOURCE_RELIABILITY.items()}
        rt = {k: min(1.0, x * scale) for k, x in C.TEXT_SOURCE_RELIABILITY.items()}
        by_reliability.append(sens_row(f"reliabilities x{scale} (capped at 1.0)",
                                       [fuse(v, t, vision_reliability=rv, text_reliability=rt).priority_score
                                        for v, t in ev]))

    result = {
        "generated_at": now_iso(),
        "n_incidents": len(incidents),
        "incidents_generated_at": data["generated_at"],
        "default": {"band_counts": {b: full_bands.count(b) for b in ["Critical", "High", "Moderate", "Low"]},
                    "top10_ids": [incidents[i]["id"] for i in top_idx]},
        "ablation": ablation,
        "sensitivity": {"vision_weight": by_weight, "reliability_scale": by_reliability},
        "notes": [
            "No ground-truth priority exists; these numbers describe behaviour and stability, not accuracy.",
            "Kendall tau is tau-b (ties handled) between the variant's and the full rule's priority scores.",
            "Scaling every reliability by the same factor leaves priorities unchanged (it cancels in the ratio); "
            "only the 1.0 cap on x1.2 changes relative influence, which is why x0.8 gives tau = 1.",
            "Weather is identical for every incident in the city, so (d) shifts all scores together; its "
            "band changes show what adding it would do, not a better ranking.",
        ],
    }
    out = MODELS_DIR / "fusion_analysis.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
