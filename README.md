# MDRIS — Flood Response Decision Support

MDRIS helps flood-response authorities (SDMA, NDRF, district collectors, field officers) answer one question:
**which incident do I send a team to first?** Applied Deep Learning course project.

- A **fine-tuned YOLOv8n** detects people and vehicles in danger in flood imagery.
- A **fine-tuned DistilBERT** classifies incident text urgency as Low / Medium / High.
- A documented **confidence × source-reliability weighted fusion rule** combines them into a priority score, a risk
  band, a separate **evidence strength**, a **conflict flag** and a suggested action.
- **Weather is shown for context only and never enters the score.**

The full specification is `docs/MDRIS_CONTEXT.md`; results are in `docs/EVALUATION.md`; data provenance in `DATA.md`.

## What is fine-tuned, on what, with which results

| Model | Base | Fine-tuned on | Test result | Baselines (same test split) |
|---|---|---|---|---|
| Vision | YOLOv8n (COCO) | "detecting humans in floods" (GNOSEF, Roboflow, v2, public domain): 4,152 train / 507 val / 387 test images | **mAP50 0.384**, mAP50-95 0.181 (only class present: `person`, P 0.629 / R 0.330) | COCO YOLOv8n without fine-tuning: person AP50 0.137 vs 0.365 fine-tuned (same scoring code) |
| Urgency NLP | distilbert-base-uncased | HumAID flood tweets (Kerala 2018, Midwest US 2019, Maryland 2018, Sri Lanka 2017; CC BY-NC 4.0): 7,816 / 1,675 / 1,675, labels mapped to Low/Medium/High | **macro-F1 0.771**, accuracy 0.805 (F1 Low 0.779, Medium 0.849, High 0.684) | TF-IDF + LR 0.749, zero-shot NLI 0.251, majority 0.241 (macro-F1) |

Fusion: 12 unit tests (T1–T8 from the spec) pass; ablation over the 40 demo incidents shows that dropping
renormalization changes 9 of 40 risk bands (Kendall τ 0.69), and the top-10 ranking stays at τ ≥ 0.82 across
vision weights 0.4–0.8.

All numbers are on held-out test splits; see `docs/EVALUATION.md` for per-class tables, baselines, figures and
failure cases, and the Model Evaluation page in the app.

## What is simulated

The demo shows **40 simulated incidents** (`backend/data/incidents.json`, seed 42). Their **images** are real images
from the vision test split and their **texts** are real held-out HumAID tweets (mentions and URLs scrubbed). Every
model output and fusion result was produced by the real fine-tuned models through the backend's own functions.
**Locations** (Chennai-area localities with jitter), **timestamps** and the **pairing** of images with texts are
synthetic. Every API response carries `data_origin` (`model` | `simulated` | `static_demo` | `live`) and every
screen shows it as a badge; the Resources page is a fixed static demo.

## Fusion rule (section 8)

```
S_v = 1 − Π(1 − w_c · conf)          noisy-OR over detections (conf ≥ 0.25)
S_n = Σ p_k · value_k                 expected urgency (Low 0.15, Medium 0.50, High 0.90)
eff = modality weight (0.6 / 0.4) × source reliability × confidence
priority = 100 · Σ(eff · S) / Σ eff   evidence = Σ eff   conflict = |S_v − S_n| > 0.5
```

Missing modalities are renormalized away (not scored as zero). All constants are documented design choices, not
calibrated values; `backend/ai/fusion/constants.py` is the single source of truth. Tests T1–T8 in
`backend/tests/test_fusion.py` assert the exact expected values from the spec.

## LLM second opinion (not a trained model)

The Vision page and Incident Analyzer can ask an external LLM (Groq, `qwen/qwen3.8-27b`, set by `GROQ_API_KEY` /
`GROQ_VISION_MODEL` in `backend/.env`) to list the people and vehicles it sees. It is labelled
**EXTERNAL LLM** everywhere (`data_origin: "external_llm"`), returns counts and descriptions only (no invented
confidence), flags when it disagrees with YOLOv8n, and is **never used in the priority score**. It exists because the
fine-tuned YOLOv8n has low recall on clean photos (test recall 0.33).

## Design exclusions

Flood only. No database, authentication, Docker, WebSockets or live feeds. Satellite imagery is not a vision source
(the detector is trained on ground-level photos). Weather is display-only. Fusion is not learned (no labelled
fusion data exists).

## How to run

Requirements: Python with PyTorch installed first (CUDA build to train; CPU is enough to run the backend), Node 20+.
This machine uses `.venv-gpu` (Python 3.13 + the system CUDA torch); see `docs/PROGRESS.md` for why.

```powershell
# Backend (http://localhost:8000, docs at /docs)
cd backend
..\.venv-gpu\Scripts\python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (http://localhost:5173)
cd frontend
npm install
npm run dev

# Tests
cd backend
..\.venv-gpu\Scripts\python -m pytest -q
```

Reproduce the models (datasets are not in git; see `DATA.md`):

```powershell
python training/prepare_vision.py
python training/prepare_text.py
python -u training/train_yolo.py   > training/runs/yolo_train.log 2>&1
python training/eval_vision.py
python -u training/train_urgency.py > training/runs/urgency_train.log 2>&1
python training/eval_urgency.py
python training/generate_incidents.py
python training/fusion_analysis.py
python training/write_evaluation.py
```

Training scripts refuse to run without CUDA. Train the two models one after the other (4 GB GPU).

## Limitations

- **The vision dataset export is single-class** (`person` only, GNOSEF v2): the detector cannot output
  `person_in_water`, `person_in_vehicle` or `vehicle`, so vision scores are capped by the `person` weight (0.45 per
  detection). A 4-class export would work with the pipeline unchanged.
- Roboflow's validation split overlaps its training split; only test metrics are reported.
- **Urgency labels are mapped from HumAID information-type categories**, not human urgency judgements; Medium
  dominates (57%).
- Domain shift: mostly non-Indian imagery; English tweets vs Tamil/Tanglish reports; Kerala is 71% of the text data.
- Fusion weights and reliabilities are uncalibrated design choices; there is no ground-truth priority.
- HumAID is CC BY-NC 4.0 and its terms ask that tweet contents be kept confidential: `incidents.json` contains 34
  tweet texts, so check before pushing this repository to a public remote.

## Repository layout

`backend/` FastAPI (routers, `ai/` models + fusion, `data/incidents.json`, `tests/`) · `frontend/` React 19 + Vite +
TypeScript + Tailwind 4 · `training/` data prep, training, evaluation, incident generator · `models/` checkpoints,
`metrics.json`, figures · `docs/` spec, evaluation, progress log.
