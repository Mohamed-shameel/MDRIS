"""
Generate the 40 simulated demo incidents (docs/MDRIS_CONTEXT.md section 10)
=============================================================================
Real held-out inputs, real models, synthetic places and times:
- images from the vision TEST split, texts from the NLP TEST split (never trained on);
- every prediction comes from the backend's own functions
  (ai.vision.predict.predict, ai.nlp.predict.predict, ai.fusion.fusion_engine.fuse);
- locations are flood-prone Chennai-area localities with +-0.005 deg jitter and
  timestamps are spread over the 6 hours before generation. Both are labelled simulated.

Mix: 70% image + text, 15% image only, 15% text only. Of the paired incidents,
60% are concordant (image S_v tercile matched to the predicted urgency level) and
40% are random pairs, which naturally produces some conflicts.

Writes backend/data/incidents.json and backend/static/incidents/SIM-XXXX.jpg.
Needs both fine-tuned models; if either is missing it stops (no fallback incidents).

Usage:  python training/generate_incidents.py
"""

import json
import random
import shutil
from datetime import datetime, timedelta

import pandas as pd

from common import PROCESSED_TEXT_DIR, PROCESSED_VISION_DIR, SEED, URGENCY_LABELS, now_iso, set_seed

N_INCIDENTS = 40
MIX = {"image_text": 0.70, "image_only": 0.15, "text_only": 0.15}
CONCORDANT_SHARE = 0.60
IMAGE_POOL = 150
TEXT_POOL = 300
HOURS_BACK = 6
JITTER_DEG = 0.005

VISION_SOURCE_MIX = {"officer_photo": 0.30, "drone": 0.15, "cctv": 0.20, "citizen_upload": 0.35}
TEXT_SOURCE_MIX = {"officer_report": 0.25, "control_room_call": 0.25, "citizen_sms": 0.30, "social_media": 0.20}

LOCATIONS = [
    ("Velachery", "Chennai", 12.9815, 80.2180),
    ("Pallikaranai", "Chengalpattu", 12.9376, 80.2139),
    ("Tambaram", "Chengalpattu", 12.9249, 80.1000),
    ("Mudichur", "Chengalpattu", 12.9110, 80.0700),
    ("Perungudi", "Chennai", 12.9654, 80.2461),
    ("Adyar", "Chennai", 13.0012, 80.2565),
    ("Kotturpuram", "Chennai", 13.0180, 80.2420),
    ("Saidapet", "Chennai", 13.0213, 80.2231),
    ("KK Nagar", "Chennai", 13.0412, 80.1994),
    ("Mylapore", "Chennai", 13.0339, 80.2619),
    ("Ashok Nagar", "Chennai", 13.0359, 80.2123),
    ("Manali", "Chennai", 13.1667, 80.2667),
]


def pick(rng: random.Random, mix: dict[str, float]) -> str:
    return rng.choices(list(mix), weights=list(mix.values()))[0]


def main() -> None:
    set_seed(SEED)
    rng = random.Random(SEED)

    import config
    from ai import registry
    from ai.fusion.fusion_engine import fuse
    from ai.fusion.schemas import TextEvidence, VisionEvidence
    from ai.nlp.predict import predict as nlp_predict
    from ai.vision.predict import predict as vision_predict
    from PIL import Image

    vision_state, nlp_state = registry.load_vision(), registry.load_nlp()
    if not vision_state.loaded or not nlp_state.loaded:
        raise SystemExit(f"STOP: both models are needed. vision: {vision_state.error} | nlp: {nlp_state.error}")

    # ── Candidate pools (test splits only) ──────────────────────────────────
    test_images = sorted((PROCESSED_VISION_DIR / "test" / "images").iterdir())
    image_pool = rng.sample(test_images, min(IMAGE_POOL, len(test_images)))
    image_scores = []
    for p in image_pool:
        with Image.open(p) as im:
            res, _ = vision_predict(im.convert("RGB"), "officer_photo")
        image_scores.append((p, res["vision_score"]))
    ranked = sorted(image_scores, key=lambda x: (x[1], x[0].name))
    third = len(ranked) / 3
    image_tercile = {p: min(2, int(i // third)) for i, (p, _) in enumerate(ranked)}  # 0 low .. 2 high

    test_text = pd.read_csv(PROCESSED_TEXT_DIR / "test.csv")
    text_rows = test_text.sample(n=min(TEXT_POOL, len(test_text)), random_state=SEED).to_dict("records")
    for row in text_rows:
        row["pred_level"] = URGENCY_LABELS.index(nlp_predict(row["text"], "citizen_sms")["urgency"])

    # ── Plan the 40 incidents ───────────────────────────────────────────────
    n_both = round(N_INCIDENTS * MIX["image_text"])
    n_img = round(N_INCIDENTS * MIX["image_only"])
    n_txt = N_INCIDENTS - n_both - n_img
    n_conc = round(n_both * CONCORDANT_SHARE)

    free_images = [p for p, _ in image_scores]
    free_texts = list(range(len(text_rows)))
    rng.shuffle(free_images)
    rng.shuffle(free_texts)
    plans = []
    for _ in range(n_conc):
        levels = [lvl for lvl in range(3)
                  if any(image_tercile[p] == lvl for p in free_images)
                  and any(text_rows[t]["pred_level"] == lvl for t in free_texts)]
        lvl = rng.choice(levels)
        img = next(p for p in free_images if image_tercile[p] == lvl)
        txt = next(t for t in free_texts if text_rows[t]["pred_level"] == lvl)
        free_images.remove(img)
        free_texts.remove(txt)
        plans.append(("concordant", img, txt))
    for _ in range(n_both - n_conc):
        plans.append(("random", free_images.pop(), free_texts.pop()))
    for _ in range(n_img):
        plans.append(("image_only", free_images.pop(), None))
    for _ in range(n_txt):
        plans.append(("text_only", None, free_texts.pop()))
    rng.shuffle(plans)

    # ── Run the real models and fuse ───────────────────────────────────────
    now = datetime.now().astimezone().replace(microsecond=0)
    times = sorted((now - timedelta(seconds=rng.randint(0, HOURS_BACK * 3600)) for _ in plans))
    out_dir = config.STATIC_INCIDENTS_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    incidents = []
    for i, ((pairing, img_path, txt_idx), created) in enumerate(zip(plans, times), 1):
        inc_id = f"SIM-{i:04d}"
        name, district, lat, lon = rng.choice(LOCATIONS)
        location = {"name": name, "district": district,
                    "lat": round(lat + rng.uniform(-JITTER_DEG, JITTER_DEG), 5),
                    "lon": round(lon + rng.uniform(-JITTER_DEG, JITTER_DEG), 5)}

        image = vision = None
        if img_path is not None:
            source = pick(rng, VISION_SOURCE_MIX)
            with Image.open(img_path) as im:
                vision, annotated = vision_predict(im.convert("RGB"), source)
            annotated.save(out_dir / f"{inc_id}.jpg", quality=88)
            url = f"/static/incidents/{inc_id}.jpg"
            vision["annotated_image_url"] = url
            image = {"url": url, "source": source,
                     "dataset_file": f"test/images/{img_path.name}"}

        text = nlp = None
        if txt_idx is not None:
            row = text_rows[txt_idx]
            source = pick(rng, TEXT_SOURCE_MIX)
            nlp = nlp_predict(row["text"], source)
            text = {"content": row["text"], "source": source,
                    "humaid_label": row["humaid_label"], "event": row["event"]}

        fusion = fuse(
            VisionEvidence(detections=[{"class_name": d["class_name"], "confidence": d["confidence"]}
                                       for d in vision["detections"]], source=vision["source"]) if vision else None,
            TextEvidence(probabilities=nlp["probabilities"], source=nlp["source"]) if nlp else None,
        ).model_dump()

        incidents.append({
            "id": inc_id,
            "simulated": True,
            "created_at": created.isoformat(),
            "pairing": pairing,
            "location": location,
            "image": image,
            "text": text,
            "vision": vision,
            "nlp": nlp,
            "fusion": fusion,
        })

    payload = {
        "generated_at": now_iso(),
        "seed": SEED,
        "simulated": True,
        "model_versions": {
            "vision": registry.model_info(registry.VISION),
            "nlp": registry.model_info(registry.NLP),
        },
        "design": {
            "count": N_INCIDENTS, "mix": MIX, "concordant_share_of_paired": CONCORDANT_SHARE,
            "vision_source_mix": VISION_SOURCE_MIX, "text_source_mix": TEXT_SOURCE_MIX,
            "image_pool": "vision test split", "text_pool": "NLP test split",
            "timestamps": f"uniform over the {HOURS_BACK} h before generation",
            "locations": f"Chennai-area localities with +-{JITTER_DEG} deg jitter",
        },
        "incidents": incidents,
    }
    config.INCIDENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.INCIDENTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    bands = pd.Series([x["fusion"]["risk_band"] for x in incidents]).value_counts().to_dict()
    conflicts = sum(x["fusion"]["conflict"] for x in incidents)
    print(f"Wrote {len(incidents)} incidents to {config.INCIDENTS_PATH}")
    print(f"Pairing: both={n_both} (concordant {n_conc}), image only={n_img}, text only={n_txt}")
    print(f"Bands: {bands} | conflicts: {conflicts}")


if __name__ == "__main__":
    main()
