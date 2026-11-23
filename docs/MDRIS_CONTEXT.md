# MDRIS — Complete Project Context (v2, all decisions locked)

**Multimodal Flood Response Intelligence System**
Applied Deep Learning course project (5th semester), built with a teammate.

> **Version:** 2.0 · **Date:** 29 Sep 2026
> **Supersedes:** `MDRIS_MASTER_CONTEXT.md` (v1). Every item that v1 marked `OPEN` or `PROPOSED` is now decided.
> **Audience:** the developer, the teammate, and Claude Code (which reads this file together with `CLAUDE.md` in the repo root).
> **Scope lock:** flood response only.
> **Place this file at:** `docs/MDRIS_CONTEXT.md` in the repo.

---

## 0. How to use this document

- **Sections 1–4** say what the system is and why it is shaped this way. Read them once.
- **Sections 5–13** are the build specification: datasets, repo layout, training, fusion maths, API, UI. Claude Code implements exactly this.
- **Section 14** is the runbook: module order, done-when checks, time budget, and prompts for Claude Code.
- **Sections 15–17** cover evaluation, viva defence and risks.

**Rule:** if the code and this document disagree, fix the code, or change this document first and then the code. Never let them silently drift.

---

## 1. One-paragraph summary

MDRIS is a decision-support dashboard for flood-response authorities (Tamil Nadu SDMA, NDRF, district collectors, field officers). A **fine-tuned YOLOv8n** detects people and vehicles in danger in flood imagery. A **fine-tuned DistilBERT** classifies the urgency of incident text as Low, Medium or High. A **confidence- and source-reliability-weighted fusion rule** combines the two into a ranked priority score with a separate **evidence strength** value and a **conflict flag**. Weather is shown for context only and is never part of the score. All incidents in the demo are **simulated**: they are real dataset images and real held-out crisis tweets, run through the real models, placed on a Chennai map and labelled as simulated everywhere. The teammate's FastAPI + React shell is reused. Everything fake inside it is replaced or labelled.

---

## 2. Problem and users

Flood response in Tamil Nadu is split across channels: IMD weather, ISRO satellite products, police road status, control-room distress calls and field-officer reports. Authorities combine these by phone, radio and spreadsheet. The question they cannot answer quickly is **"which incident do I send a team to first?"**

| | |
|---|---|
| **Users** | Disaster-management authorities: SDMA, NDRF, district collectors, field officers |
| **Not users** | Citizens. Citizens appear only as information *sources* (a photo or an SMS), never as operators |
| **Core output** | A priority queue of incidents, each with a score, a risk band, how strong the evidence is, whether the signals disagree, and a suggested action |

---

## 3. Design principles (locked)

1. **Deep learning is the graded substance.** Two fine-tuned models, each evaluated against baselines.
2. **The fusion rule is the research contribution.** It weights by confidence and source reliability, renormalizes over the evidence actually present, and reports evidence strength and conflict separately from the score.
3. **Honesty over polish.** Every number on screen is either computed by a real model or labelled as simulated or static. No invented telemetry.
4. **Depth over breadth.** Flood only. No database, auth, Docker or live feeds this semester.
5. **Reproducible.** Fixed seeds, scripted data preparation, and a `metrics.json` saved next to every trained model.

---

## 4. Decision register (all locked)

| ID | Decision | Rationale |
|---|---|---|
| D-01 | Scope: flood only | A semester-sized project needs depth |
| D-02 | Users are authorities, not citizens | Matches the real coordination problem |
| D-03 | Two fine-tuned DL models: vision and NLP | Applied Deep Learning course |
| D-04 | Vision model: YOLOv8n object detection, fine-tuned | Produces concrete classes and boxes that fusion can score; `ultralytics` and `yolov8n.pt` are already in the repo. SegFormer is dropped. |
| D-05 | Vision dataset: Roboflow **"detecting humans in floods" (GNOSEF)**; fallback: **"final" (t5)** | Flood scenes with people and vehicles in danger (Section 5.1) |
| D-06 | Four canonical vision classes: `person_in_water`, `person_in_vehicle`, `person`, `vehicle` | Dataset-independent; both candidate datasets map onto them |
| D-07 | NLP model: `distilbert-base-uncased`, fine-tuned for **3-level urgency** (Low/Medium/High) | 3 classes stay robust on about 10k rows; a documented mapping from HumAID categories |
| D-08 | NLP dataset: **HumAID**, flood events only, all available flood rows (not subsampled to 2–3k) | More data, and it includes the 2018 Kerala floods (an Indian event) |
| D-09 | Urgency mapping from HumAID categories: Section 5.2.3 | Documented and defensible; its limitations are stated |
| D-10 | Fusion: two inputs, confidence × reliability × modality weight, renormalized | The research contribution |
| D-11 | Modality weights `VISION_WEIGHT = 0.6`, `NLP_WEIGHT = 0.4` as named constants | Direct visual evidence is stronger; constants are documented and tunable |
| D-12 | Vision score by **noisy-OR** over class-weighted detections | Grows with more evidence but saturates below 1; many cars cannot outweigh one person in water |
| D-13 | Satellite is not a vision source | The detector is trained on ground-level photos; satellite tiles are out of domain |
| D-14 | Weather is display-only, never in the score | City-level rainfall does not measure incident severity |
| D-15 | Missing modality: renormalize over present inputs; both missing → HTTP 422 | Fixes the old "absent module lowers the score" bug |
| D-16 | Outputs: `priority_score`, `risk_band`, `evidence_strength`, `conflict`, `dominant_class`, `suggested_action`, full `breakdown` | Commanders need to know how sure the system is, not just the score |
| D-17 | Action map keyed on **dominant vision class × risk band**, with text-only rows keyed on band | Every action traces to a real model output |
| D-18 | Demo incidents: **precomputed** by an offline generator using the real models; 40 incidents; seed 42 | Deterministic demo, no GPU load at runtime, honest |
| D-19 | Incident text comes from the **HumAID test split** (never trained on), with @mentions and URLs scrubbed | Real, held-out model input; no train leakage |
| D-20 | Incident locations: Chennai-area flood-prone localities with jitter, labelled simulated | Tamil Nadu framing without claiming real events |
| D-21 | Every API payload carries `data_origin`: `model`, `simulated`, `static_demo` or `live` | Honesty is machine-checkable |
| D-22 | `/api/analytics` is replaced by `/api/evaluation`, serving real model metrics; the Analytics page becomes **Model Evaluation** | Turns a mock page into the strongest viva page |
| D-23 | System Status reports real load state, checkpoint names, training dates and metrics | Removes the invented "NLP 99%" |
| D-24 | Resources page kept, with a permanent "Static demo" badge | Out of scope; the badge keeps it honest |
| D-25 | New endpoint `POST /api/incident/analyze` runs vision + NLP + fusion in one call | The Incident Analyzer page needs one round trip |
| D-26 | Sentiment pipeline removed | Dead signal; never used by fusion |
| D-27 | The zero-shot pipeline is kept only as an **evaluation baseline** in `training/`, not in the backend | A before/after comparison for the report |
| D-28 | Baselines: majority class, TF-IDF + logistic regression, zero-shot NLI | Shows what fine-tuning adds |
| D-29 | Fusion evaluated through fixed test cases, an ablation and a weight sensitivity analysis | No ground-truth priority exists, so show behaviour and stability |
| D-30 | Single agent (Claude Code), sequential modules, one commit per module, branch `rebuild` | Simple, one shared GPU, clean history |
| D-31 | Training settings sized for **RTX 2050, 4 GB VRAM** (Section 7) | Avoid out-of-memory failures |
| D-32 | `datasets/` and large weights are git-ignored; `metrics.json`, figures, the YOLO `best.pt` (~6 MB) and `incidents.json` are committed | The repo stays small but the results are reviewable |
| D-33 | `mdris-mvp/` Streamlit scaffold deleted | Duplicate track |
| D-34 | Backend device: `DEVICE=auto` (CUDA if available, else CPU) | Runs on any machine for the demo |
| D-35 | Tests: pytest for fusion (8 cases) and an API smoke test; `npm run build` must pass | A minimum quality bar |
| D-36 | **Added 2026-09-30 (user-approved):** an LLM vision **second opinion** via the Groq API (`POST /api/vision/llm-check`, `data_origin: "external_llm"`). Shown beside the YOLO result on the Vision page and Incident Analyzer, always labelled with the LLM's name. **Never enters fusion**, never labelled as YOLO. No invented confidence: it returns per-class counts and descriptions only. Missing `GROQ_API_KEY` → 503; Groq failure or invalid reply → 502 | The fine-tuned YOLO has low recall on clean photos (test R 0.33); a visible second opinion exposes misses without replacing the graded model |
| D-37 | **Changed 2026-09-30 (user decision):** the backend **serves the COCO-pretrained YOLOv8n** (`backend/yolov8n.pt`), selected by `VISION_MODEL_VARIANT=coco_pretrained` (default; `finetuned` serves `models/vision/best.pt`). COCO ids are mapped to canonical classes by `COCO_TO_CANONICAL` in `constants.py` (person → `person`; car, motorcycle, bus, truck → `vehicle`; all other COCO classes ignored). It is labelled **"YOLOv8n (COCO-pretrained)"** everywhere, never "fine-tuned"; its headline metric is its own test-split baseline (person AP50 0.137). The fine-tuned model stays trained, evaluated and shown on Model Evaluation for comparison | On clean photos the fine-tuned model missed people the COCO model finds (user image: 0 vs 5 detections); on the dataset's own test frames COCO is weaker (0.137 vs 0.365). The trade-off is stated, not hidden |

---

## 5. Datasets

### 5.1 Vision

**Primary:** "detecting humans in floods" by GNOSEF
- URL: https://universe.roboflow.com/gnosef/detecting-humans-in-floods-gukif
- About 2,020 images. Raw classes: `car`, `drowning`, `person`, `in-car`.
- Licence: shown as CC0 on the Roboflow page. **Confirm** in the `README.roboflow.txt` inside the export and record it in `DATA.md`.

**Fallback** (use only if most primary images turn out to be swimming pools rather than floods): "final" by t5
- URL: https://universe.roboflow.com/t5-rkshu/final-muucf
- About 2,270 images. Raw classes: `car`, `not-drowning`, `stuck-in-car`, `stuck-in-water`.

**How to download (manual, done by you):**
1. Log in to Roboflow (free account).
2. Open the dataset → **Download Dataset** → format **YOLOv8** → **Download zip to computer**.
3. Unzip so that `datasets/raw/flood_yolo/data.yaml` exists, with `train/`, `valid/` and usually `test/` beside it.

#### 5.1.1 Canonical class remap (locked)

`training/prepare_vision.py` rewrites every label file into these IDs. Raw names are normalized first (lowercase, trimmed, spaces and underscores become hyphens).

| Canonical ID | Canonical class | From GNOSEF | From t5 | Fusion weight `w_c` |
|---|---|---|---|---|
| 0 | `person_in_water` | `drowning` | `stuck-in-water` | **1.00** |
| 1 | `person_in_vehicle` | `in-car` | `stuck-in-car` | **0.85** |
| 2 | `person` | `person` | `not-drowning` | **0.45** |
| 3 | `vehicle` | `car` | `car` | **0.15** |

- Any raw class **not** in this table → the script **stops** and prints the unknown names. It must not guess.
- If there is no `test/` split, carve 10% of `train/` into `test/` by image (seed 42) and record that in `DATA.md`.
- Output: `datasets/processed/flood_yolo/` with its own `data.yaml` (absolute paths, `names` = the four canonical classes).
- Print and save to `DATA.md`: images per split, instances per class per split, and any class with fewer than 100 training instances.

### 5.2 Text (NLP)

**Source:** HumAID (Alam et al., ICWSM 2021), CrisisNLP / QCRI
- Page: https://crisisnlp.qcri.org/humaid_dataset
- Direct download (event-wise set 1): https://crisisnlp.qcri.org/data/humaid/HumAID_data_events_set1_47K.tar.gz
- Full set (use if Kerala is not in set 1): https://doi.org/10.7910/DVN/A7NVF7
- Licence: CC BY-NC 4.0, non-commercial. Fine for academic use. Cite the paper.
- **Do not use** the Hugging Face copy `QCRI/HumAID-all`: it has no event column, so it cannot be filtered to floods.

**Flood events in HumAID (annotated tweets):**

| Event | Annotated tweets |
|---|---|
| 2018 Kerala Floods (India) | 8,056 |
| 2019 Midwestern U.S. Floods | 1,930 |
| 2018 Maryland Floods | 747 |
| 2017 Sri Lanka Floods | 575 |
| **Total available** | **~11,300** |

Extract into `datasets/raw/humaid/`. `prepare_text.py` must **inspect the folder first** (file names, columns) and not assume a layout. It selects every event folder whose name contains `flood`.

#### 5.2.1 Cleaning (locked order)
1. Load all splits of all flood events and keep columns `tweet_id`, `tweet_text`, `class_label`, plus `event` (from the folder name).
2. Drop rows labelled `dont_know_cant_judge` (if present) and empty texts.
3. Normalize: strip a leading `RT @x:`, replace URLs with `URL`, replace `@mentions` with `@USER`, collapse whitespace. Scrubbing mentions also protects privacy in the demo.
4. **Deduplicate on normalized text** before splitting. Retweets would otherwise leak between train and test.
5. Map to urgency (5.2.3).
6. Stratified split **70 / 15 / 15** (train/val/test) on urgency, seed 42.
7. Save `datasets/processed/flood_text/{train,val,test}.csv` with columns `text, urgency, humaid_label, event`.
8. If fewer than 1,500 rows remain, **stop and report**.

#### 5.2.2 Why re-split instead of using HumAID's splits
The labels are remapped and duplicates removed, so the official splits no longer apply cleanly. This is stated in `DATA.md`.

#### 5.2.3 Urgency mapping (locked)

| Urgency | HumAID categories | Reasoning |
|---|---|---|
| **High** | `requests_or_urgent_needs`, `injured_or_dead_people`, `missing_or_found_people`, `displaced_people_and_evacuations` | Unmet needs or people directly at risk; needs a response now |
| **Medium** | `infrastructure_and_utility_damage`, `caution_and_advice`, `rescue_volunteering_or_donation_effort` | Situational or operational information; response already under way or planning needed |
| **Low** | `sympathy_and_support`, `other_relevant_information`, `not_humanitarian` | No operational action required |

**Stated limitation:** HumAID labels describe *information type*, not urgency. `rescue_volunteering_or_donation_effort` mixes ongoing rescues with donation drives, and it is the largest Kerala category, so Medium will dominate. This is handled with class-weighted loss and macro-F1, and it is said openly in the report.

---

## 6. Repository layout (target)

```
MDRIS/
├── CLAUDE.md                     # guardrails for Claude Code (short)
├── README.md                     # rewritten in M6
├── DATA.md                       # dataset provenance, licences, splits, stats, biases
├── .gitignore                    # datasets/, models/nlp/*.safetensors, training/runs/, __pycache__, node_modules
├── docs/
│   ├── MDRIS_CONTEXT.md          # this file
│   └── EVALUATION.md             # generated tables + figure links (M2, M3, M4)
├── datasets/                     # git-ignored
│   ├── raw/{flood_yolo,humaid}/
│   └── processed/{flood_yolo,flood_text}/
├── training/
│   ├── requirements.txt
│   ├── common.py                 # REPO_ROOT, seeds, paths, label maps (single source of truth)
│   ├── prepare_vision.py
│   ├── prepare_text.py
│   ├── train_yolo.py
│   ├── train_urgency.py
│   ├── eval_vision.py            # test-split metrics + failure-case images
│   ├── eval_urgency.py           # test metrics + 3 baselines
│   ├── fusion_analysis.py        # ablation + sensitivity over incidents.json
│   ├── generate_incidents.py     # builds backend/data/incidents.json
│   └── runs/                     # git-ignored ultralytics / trainer output
├── models/
│   ├── vision/  best.pt  metrics.json  figures/
│   └── nlp/     (HF files, weights git-ignored)  metrics.json  figures/
├── backend/
│   ├── main.py  config.py  .env  requirements.txt
│   ├── ai/
│   │   ├── registry.py           # loads models, exposes status + metrics
│   │   ├── vision/  model_loader.py predict.py service.py schemas.py image_utils.py
│   │   ├── nlp/     predict.py service.py schemas.py
│   │   ├── fusion/  constants.py fusion_engine.py actions.py schemas.py
│   │   └── weather/ service.py
│   ├── routers/  vision nlp fusion incident weather dashboard map alerts resources evaluation system
│   ├── data/  incidents.json  static_resources.py
│   ├── static/incidents/         # annotated images used by the demo incidents
│   ├── uploads/
│   └── tests/  test_fusion.py  test_api_smoke.py
└── frontend/                     # existing React 19 + Vite + TS + Tailwind 4 shell
```

`common.py` holds the canonical class list, the class weights, the urgency map and the seed. The backend imports the same constants from `backend/ai/fusion/constants.py`. **If a value changes, both files change in the same commit**, or `common.py` imports from the backend constants.

---

## 7. Model training specification

### 7.1 Environment (both models)

- Python 3.10 or 3.11. **CUDA PyTorch is mandatory:**
  `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121`
- `training/requirements.txt`: `ultralytics==8.2.60`, `transformers==4.42.4`, `accelerate>=0.26`, `scikit-learn`, `pandas`, `matplotlib`, `pyyaml`, `tqdm`.
- Before training, each script asserts `torch.cuda.is_available()` and **exits with an error** if it is false. No silent CPU training.
- Windows: every training script wraps its entry point in `if __name__ == "__main__":`, which multiprocessing needs.
- Seeds: 42 everywhere (`random`, `numpy`, `torch`, and the ultralytics `seed`).
- Laptop: on AC power, high-performance power profile.
- Train the two models **one after the other**. 4 GB of VRAM cannot hold both.

### 7.2 YOLOv8n fine-tune (`train_yolo.py`)

| Setting | Value | Note |
|---|---|---|
| base weights | `backend/yolov8n.pt` (COCO) | transfer learning |
| data | `datasets/processed/flood_yolo/data.yaml` | 4 classes |
| imgsz | 640 | |
| epochs | 60 | |
| patience | 15 | early stop |
| batch | 8 | drop to 4 on out-of-memory |
| workers | 2 | set 0 if the Windows dataloader hangs |
| amp | True | |
| optimizer | auto (ultralytics default) | |
| seed / deterministic | 42 / True | |
| project / name | `training/runs` / `yolov8n_flood` | `exist_ok=True` |
| freeze | 0 | **Experiment E1 (approved 2026-09-30):** `--freeze 10` freezes the backbone (layers 0–9) to keep COCO's person features. Written to run `yolov8n_flood_freeze10` and promoted to `models/vision/` only with `--promote`, after comparing test metrics |

Afterwards: copy `best.pt` to `models/vision/best.pt`; copy `results.png`, `confusion_matrix.png`, `PR_curve.png` and `F1_curve.png` into `models/vision/figures/`. Run the training as a **background process with a log file**; Claude Code polls the log rather than blocking. Expected time on an RTX 2050: about 45–90 minutes.

### 7.3 DistilBERT urgency fine-tune (`train_urgency.py`)

| Setting | Value |
|---|---|
| model | `distilbert-base-uncased` |
| labels | `{0: Low, 1: Medium, 2: High}` (order fixed in `common.py`) |
| max_length | 128 |
| batch size | 16 (train and eval) |
| learning rate | 2e-5 |
| epochs | 4, with early stopping on val macro-F1 (patience 1) |
| weight decay | 0.01 |
| warmup | 10% of steps |
| fp16 | True |
| loss | cross-entropy with **class weights = inverse class frequency** (normalized), via a custom `Trainer.compute_loss` |
| best model | `load_best_model_at_end=True`, metric `eval_macro_f1` |

Save with `save_pretrained` (model + tokenizer) to `models/nlp/`. Save the training curve as `models/nlp/figures/training_curve.png`. Expected time: about 10–20 minutes.

### 7.4 `metrics.json` schema (both models)

```json
{
  "model_name": "mdris-yolov8n-flood",
  "task": "object_detection",
  "base_model": "yolov8n.pt (COCO)",
  "trained_at": "2026-09-29T17:40:00+05:30",
  "hardware": "NVIDIA GeForce RTX 2050 4GB",
  "train_time_minutes": 63.2,
  "dataset": {"name": "detecting humans in floods (GNOSEF)", "url": "...", "licence": "CC0",
              "splits": {"train": 0, "val": 0, "test": 0}},
  "classes": ["person_in_water", "person_in_vehicle", "person", "vehicle"],
  "hyperparameters": {"imgsz": 640, "epochs_run": 0, "batch": 8, "seed": 42},
  "metrics": {"test": {"mAP50": 0.0, "mAP50_95": 0.0, "per_class": {"person_in_water": {"precision": 0.0, "recall": 0.0, "AP50": 0.0}}}},
  "baselines": {},
  "figures": ["figures/confusion_matrix.png", "figures/PR_curve.png"]
}
```

For NLP: `task: "text_classification"`, `metrics.test` = accuracy, macro-F1, weighted-F1, per-class P/R/F1, the confusion matrix as a nested list, and `per_event_macro_f1`. `baselines` holds the three baseline results (Section 15.2).

---

## 8. Fusion specification (locked)

### 8.1 Constants (`backend/ai/fusion/constants.py`)

```python
VISION_WEIGHT = 0.6
NLP_WEIGHT    = 0.4

CLASS_WEIGHTS = {            # w_c: how much one confident detection of this class implies danger
    "person_in_water":   1.00,
    "person_in_vehicle": 0.85,
    "person":            0.45,
    "vehicle":           0.15,
}

URGENCY_VALUES = {"Low": 0.15, "Medium": 0.50, "High": 0.90}

VISION_SOURCE_RELIABILITY = {
    "officer_photo":  1.00,  # trained responder, known location
    "drone":          0.90,  # operator-controlled, may be oblique
    "cctv":           0.85,  # fixed camera, low resolution, night issues
    "citizen_upload": 0.60,  # unverified, could be old or from elsewhere
}
TEXT_SOURCE_RELIABILITY = {
    "officer_report":    1.00,
    "control_room_call": 0.80,  # transcribed, second-hand
    "citizen_sms":       0.60,
    "social_media":      0.50,  # unverified, may be rumour or a repost
}

NO_DETECTION_CONFIDENCE = 0.50   # confidence assigned to "image present, nothing detected"
DETECTION_CONF_THRESHOLD = 0.25  # YOLO inference threshold
CONFLICT_THRESHOLD = 0.50        # |S_v - S_n| above this -> conflict
RISK_BANDS = [(75, "Critical"), (55, "High"), (35, "Moderate"), (0, "Low")]
EVIDENCE_LEVELS = [(0.70, "Strong"), (0.40, "Fair"), (0.0, "Weak")]
```

All values are **documented design choices, not calibrated parameters**, and the report says so. Section 15.3 tests how sensitive the rankings are to them.

### 8.2 Per-modality scores (each in 0–1)

**Vision score (noisy-OR):**
```
S_v    = 1 − Π_i (1 − w_{c_i} · conf_i)     over detections with conf ≥ 0.25
conf_v = max_i conf_i
no detections but an image was given → S_v = 0, conf_v = NO_DETECTION_CONFIDENCE
```
Why noisy-OR: each detection is independent evidence of danger. The score grows with more evidence, never exceeds 1, and one strong `person_in_water` outweighs many `vehicle`s.

**Text score (expected urgency):**
```
S_n    = Σ_k p_k · URGENCY_VALUES[k]      (softmax probabilities from DistilBERT)
conf_n = max_k p_k
```
Why expected value: it uses the whole distribution, so a text split between Medium and High lands in between instead of snapping to one class.

### 8.3 Combination

```
eff_v = VISION_WEIGHT · R_vision[src_v] · conf_v      (0 if no image)
eff_n = NLP_WEIGHT    · R_text[src_n]   · conf_n      (0 if no text)

priority_score    = 100 · (eff_v·S_v + eff_n·S_n) / (eff_v + eff_n)
evidence_strength = eff_v + eff_n                     (in 0–1 because the weights sum to 1)
conflict          = both present AND |S_v − S_n| > 0.5
risk_band         = first band in RISK_BANDS with priority_score ≥ threshold
dominant_class    = argmax over detections of w_c · conf (or "text_only" if there are none)
both missing      → raise ValueError → HTTP 422
```

**Key property to explain in the viva:** reliability and confidence change **how much a source influences** the score, not the score directly. A less reliable source moves the score *toward the other source* and lowers `evidence_strength`. It does not blindly push the score down. Test T5 below demonstrates this.

### 8.4 Worked test cases (these become `backend/tests/test_fusion.py`)

The numbers were computed with the formulas above; tests assert them to ±0.5.
`hi_text = {Low: .05, Medium: .10, High: .85}`, `lo_text = {Low: .85, Medium: .10, High: .05}`.

| # | Scenario | Inputs | Expected |
|---|---|---|---|
| T1 | Both strong, officer sources | officer_photo: person_in_water 0.90 + officer_report: hi_text | priority **87.0**, **Critical**, evidence **0.88**, no conflict |
| T2 | Vision only | officer_photo: person_in_water 0.90 | priority **90.0**, **Critical**, evidence **0.54** (renormalized, not capped at 60) |
| T3 | Text only | officer_report: hi_text | priority **82.2**, **Critical**, evidence **0.34** |
| T4 | Conflicting signals | officer_photo: person_in_water 0.90 + officer_report: lo_text | priority **63.8**, **High**, evidence 0.88, **conflict = true** |
| T5a | Reliability, officer photo | officer_photo: person_in_vehicle 0.80 + citizen_sms: lo_text | priority **54.4**, Moderate, eff_v **0.48**, evidence **0.684** |
| T5b | Same content, citizen photo | citizen_upload: person_in_vehicle 0.80 + citizen_sms: lo_text | priority **49.0**, Moderate, eff_v **0.288**, evidence **0.492**. Assert eff_v and evidence are lower than T5a |
| T6 | Image with no detections + urgent text | officer_photo: [] + officer_report: hi_text | priority **43.7**, **Moderate**, **conflict = true** (flag for ground verification) |
| T7 | Many vehicles, no people | officer_photo: 5 × vehicle 0.80 | S_v **0.472**, priority **47.2**, **Moderate**; must not reach High |
| T8 | Nothing provided | none | raises / HTTP **422** |

### 8.5 Action map (`backend/ai/fusion/actions.py`)

| dominant_class ↓ / band → | Critical | High | Moderate | Low |
|---|---|---|---|---|
| `person_in_water` | Dispatch rescue boat and NDRF swift-water team immediately | Dispatch rescue boat; alert nearest NDRF unit | Send field officer to verify; boat on standby | Log and monitor |
| `person_in_vehicle` | Dispatch rescue team for vehicle extraction; close the road | Dispatch rescue team; issue road closure | Traffic police to divert; verify occupants | Log; issue traffic advisory |
| `person` | Evacuate to nearest relief camp; send boat if access is cut | Organise evacuation; send relief supplies | Field officer check; alert relief camp | Monitor |
| `vehicle` | Close road, divert traffic, check for occupants | Close road and divert traffic | Issue traffic advisory | Monitor |
| `text_only` | Call back reporter; dispatch verification team now | Call back reporter; send field officer | Verify through field officer | Log |

If `conflict` is true, append: **"Signals disagree: verify on ground before committing resources."**

### 8.6 What fusion does not do
- It does not use weather.
- It is not learned; no labelled fusion data exists.
- It does not claim to be calibrated. It is a documented, auditable decision rule.

---

## 9. Backend specification

### 9.1 Configuration (`backend/config.py` + `.env`)

`config.py` resolves `REPO_ROOT = Path(__file__).resolve().parent.parent` and reads:

```
VISION_MODEL_PATH=models/vision/best.pt       # relative to REPO_ROOT
NLP_MODEL_DIR=models/nlp
INCIDENTS_PATH=backend/data/incidents.json
UPLOAD_FOLDER=backend/uploads
DEVICE=auto                                   # auto | cuda | cpu
OPENWEATHER_KEY=                              # optional, https://openweathermap.org/api (free tier)
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```
`MODEL_PATH` is renamed to `VISION_MODEL_PATH`.

### 9.2 Startup behaviour
- Lifespan loads the YOLO and DistilBERT models once through `ai/registry.py`.
- If a model fails to load: log at **ERROR** with the path and the exception, mark it `loaded: false` in the registry, and have that endpoint return **HTTP 503** `{"detail": "vision model not loaded: <reason>"}`. **Never return a silent fallback result.**
- Remove `is_fallback`, `pending-flood-segformer`, the zero-shot pipeline, the sentiment pipeline and any COCO→disaster mapping.
- Serve static paths: `/uploads`, `/static/incidents`, and `/figures/vision` and `/figures/nlp` (mapped to `models/*/figures`).

### 9.3 Endpoints (final contract)

Every response includes `"data_origin"`.

| Method | Path | data_origin | Purpose |
|---|---|---|---|
| GET | `/health` | live | `{"status": "ok"}` |
| POST | `/api/vision/predict` | model | multipart `file` (JPEG/PNG, ≤15 MB), optional `source` |
| GET | `/api/vision/status` | live | load state and checkpoint |
| POST | `/api/nlp/analyze` | model | `{"text": str (3–2000 chars), "source"?: str}` |
| GET | `/api/nlp/labels` | live | urgency labels + the HumAID mapping table |
| POST | `/api/fusion/analyze` | model | fuse precomputed vision + NLP outputs |
| POST | `/api/incident/analyze` | model | multipart: optional `image`, optional `text`, `image_source`, `text_source`; runs all three |
| GET | `/api/incidents` | simulated | all demo incidents, sorted by priority desc then evidence desc |
| GET | `/api/incidents/{id}` | simulated | one incident |
| GET | `/api/dashboard` | simulated | KPIs computed from incidents.json |
| GET | `/api/map` | simulated | GeoJSON-like list of incidents |
| GET | `/api/alerts` | simulated | incidents with band High or Critical |
| GET | `/api/resources` | static_demo | unchanged static data |
| GET | `/api/evaluation` | model | both `metrics.json` files + figure URLs + fusion analysis |
| GET | `/api/system-status` | live | real registry state, device, versions, incident-file age |
| GET | `/api/weather/?lat=&lon=` | live | unchanged; without a key returns `{"configured": false}` instead of zeros |
| GET | `/api/weather/presets` | live | unchanged |
| POST | `/api/vision/llm-check` | external_llm | multipart `file` (JPEG/PNG); Groq LLM second opinion: per-class counts + descriptions (D-36). Not used by fusion |

The old `/api/analytics` is removed; the frontend switches to `/api/evaluation`.

### 9.4 Schemas

**Vision response**
```json
{
  "data_origin": "model",
  "model": {"name": "mdris-yolov8n-flood", "checkpoint": "models/vision/best.pt", "trained_at": "..."},
  "source": "officer_photo",
  "detections": [
    {"class_name": "person_in_water", "confidence": 0.87,
     "bbox": {"x1": 120, "y1": 88, "x2": 210, "y2": 260}}
  ],
  "counts": {"person_in_water": 1, "person_in_vehicle": 0, "person": 2, "vehicle": 1},
  "vision_score": 0.91,
  "vision_confidence": 0.87,
  "severity_band": "Critical",
  "annotated_image_url": "/uploads/abc123_annotated.jpg",
  "inference_ms": 41
}
```

**NLP response**
```json
{
  "data_origin": "model",
  "model": {"name": "mdris-distilbert-urgency", "path": "models/nlp", "trained_at": "..."},
  "source": "citizen_sms",
  "urgency": "High",
  "probabilities": {"Low": 0.04, "Medium": 0.13, "High": 0.83},
  "confidence": 0.83,
  "nlp_score": 0.82,
  "inference_ms": 12
}
```

**Fusion request / response**
```json
// request
{
  "vision": {"detections": [{"class_name": "person_in_water", "confidence": 0.9}], "source": "officer_photo"},
  "nlp":    {"probabilities": {"Low": 0.05, "Medium": 0.10, "High": 0.85}, "source": "officer_report"}
}
// either may be null; both null -> 422

// response
{
  "data_origin": "model",
  "priority_score": 87.0,
  "risk_band": "Critical",
  "evidence_strength": 0.88,
  "evidence_level": "Strong",
  "conflict": false,
  "single_source": false,
  "dominant_class": "person_in_water",
  "suggested_action": "Dispatch rescue boat and NDRF swift-water team immediately",
  "breakdown": {
    "S_v": 0.90, "conf_v": 0.90, "reliability_v": 1.0, "eff_v": 0.54,
    "S_n": 0.823, "conf_n": 0.85, "reliability_n": 1.0, "eff_n": 0.34,
    "weights": {"vision": 0.6, "nlp": 0.4}
  }
}
```

**Incident object** (in `incidents.json` and `/api/incidents`)
```json
{
  "id": "SIM-0007",
  "simulated": true,
  "created_at": "2026-09-29T16:12:00+05:30",
  "location": {"name": "Velachery", "district": "Chennai", "lat": 12.9791, "lon": 80.2203},
  "image": {"url": "/static/incidents/SIM-0007.jpg", "source": "cctv", "dataset_file": "test/images/xyz.jpg"},
  "text": {"content": "Water entering houses near the lake, elderly people stuck on first floor @USER URL",
           "source": "citizen_sms", "humaid_label": "requests_or_urgent_needs", "event": "kerala_floods_2018"},
  "vision": { /* vision response subset */ },
  "nlp": { /* nlp response subset */ },
  "fusion": { /* fusion response */ }
}
```
Any modality may be `null`.

---

## 10. Simulated incident generator (`training/generate_incidents.py`)

| Parameter | Value |
|---|---|
| Count | 40 |
| Seed | 42 |
| Image pool | vision **test** split only |
| Text pool | NLP **test** split only (already scrubbed) |
| Modality mix | 70% image + text, 15% image only, 15% text only |
| Pairing | 60% concordant (image S_v tercile matched to predicted-urgency level), 40% random, which naturally produces some conflicts |
| Vision sources | officer_photo 30%, drone 15%, cctv 20%, citizen_upload 35% |
| Text sources | officer_report 25%, control_room_call 25%, citizen_sms 30%, social_media 20% |
| Timestamps | spread over the 6 hours before generation time |
| Locations | the list below, with ±0.005° jitter, sampled with replacement |

It must call **the same backend functions** (`ai.vision.predict`, `ai.nlp.predict`, `ai.fusion.fusion_engine`), not reimplement them. It copies annotated images to `backend/static/incidents/` and writes `backend/data/incidents.json` with a header `{"generated_at", "seed", "model_versions", "simulated": true, "incidents": [...]}`.

**Locations** (approximate centroids of flood-prone Chennai-area localities; labelled simulated):

| Name | District | Lat | Lon |
|---|---|---|---|
| Velachery | Chennai | 12.9815 | 80.2180 |
| Pallikaranai | Chengalpattu | 12.9376 | 80.2139 |
| Tambaram | Chengalpattu | 12.9249 | 80.1000 |
| Mudichur | Chengalpattu | 12.9110 | 80.0700 |
| Perungudi | Chennai | 12.9654 | 80.2461 |
| Adyar | Chennai | 13.0012 | 80.2565 |
| Kotturpuram | Chennai | 13.0180 | 80.2420 |
| Saidapet | Chennai | 13.0213 | 80.2231 |
| KK Nagar | Chennai | 13.0412 | 80.1994 |
| Mylapore | Chennai | 13.0339 | 80.2619 |
| Ashok Nagar | Chennai | 13.0359 | 80.2123 |
| Manali | Chennai | 13.1667 | 80.2667 |

---

## 11. Frontend specification (UI decisions)

### 11.1 Keep
React 19, Vite, TypeScript, Tailwind 4, dark mode by default, the 264 px sidebar, the 80 px top bar, the gradient orbs (subtle), the typed Axios client in `services/api.ts`, Sonner toasts and the Leaflet map.

### 11.2 Navigation (final)

| Section | Label | Route | Data |
|---|---|---|---|
| Main | Command Dashboard | `/dashboard` | simulated |
| Main | Incident Map | `/dashboard/map` | simulated |
| Main | Alerts | `/dashboard/alerts` | simulated |
| Main | Resources | `/dashboard/resources` | static_demo |
| Main | Model Evaluation | `/dashboard/analytics` (route unchanged, label changed) | model |
| Main | System Status | `/dashboard/settings` | live |
| AI Modules | Vision (YOLOv8n) | `/dashboard/vision` | model |
| AI Modules | Urgency NLP (DistilBERT) | `/dashboard/nlp` | model |
| AI Modules | Incident Analyzer (Fusion) | `/dashboard/fusion` | model |
| AI Modules | Weather (context only) | `/dashboard/weather` | live |

### 11.3 Shared components (new)
- **`DataOriginBadge`**: reads `data_origin`. `simulated` shows an amber pill "SIMULATED"; `static_demo` a grey pill "STATIC DEMO"; `model` a blue pill "MODEL OUTPUT"; `live` a green pill "LIVE". Shown in the header of every data card.
- **`SimulationBanner`**: a sticky amber strip at the top of Dashboard, Map and Alerts: *"Simulated incidents: public dataset images and held-out crisis tweets processed by the real models. Locations and times are synthetic. Not live data."* It cannot be dismissed.
- **`RiskBadge`**: text plus colour, never colour alone. Critical `#DC2626`, High `#EA580C`, Moderate `#CA8A04`, Low `#16A34A`.
- **`EvidenceMeter`**: a three-segment bar with a label (Weak < 0.40 ≤ Fair < 0.70 ≤ Strong).
- **`ConflictFlag`**: an amber triangle with the tooltip "Image and text disagree: verify on ground".
- **`SingleSourceChip`**: a grey chip "Single source" when only one modality is present.
- **`FusionBreakdown`**: shows the formula with the actual numbers substituted, S_v, S_n, the reliabilities, the eff values and the weights, plus a stacked bar of vision vs text contribution. This is the explainability panel; it appears on the Incident Analyzer and in the incident drawer.
- **`IncidentDrawer`**: a right-side drawer containing the annotated image, the text with an urgency probability bar, `FusionBreakdown`, the suggested action, and source chips.

### 11.4 Pages

**Command Dashboard**
- KPI row: total incidents, Critical count, High count, conflicts needing verification, mean evidence strength.
- **Priority Queue** table (the centrepiece): rank · ID · location · RiskBadge · score · EvidenceMeter · dominant class · action (truncated) · conflict/single-source icons. Clicking a row opens `IncidentDrawer`.
- Side card: Chennai weather with the label **"Context only: not used in priority score"**. If no key is set: "Weather API key not configured".
- Remove every invented number from the old mock dashboard.

**Incident Map**
- Centre Chennai (13.0827, 80.2707), zoom 11.
- CircleMarker: radius `6 + priority/10`, fill colour from the risk band, fill opacity `0.3 + 0.6 × evidence_strength`, so weak evidence looks faded.
- Popup: ID, band, score, action and an "Open" button that opens the drawer.
- Legend: the band colours plus "faded = weak evidence".
- Filters: band checkboxes, and "conflicts only".

**Alerts**: High and Critical incidents as cards, newest first, with the band, action and conflict flag.

**Resources**: unchanged content, with a `STATIC DEMO` badge and the note "Not connected to any resource system."

**Model Evaluation** (replaces Analytics)
- Vision card: mAP50, mAP50-95, a per-class P/R/AP50 table, and the confusion matrix and PR curve images from `/figures/vision/...`, plus 6 failure-case images.
- NLP card: accuracy, macro-F1, a per-class table, the confusion matrix image, the training curve, and **a baseline comparison bar chart** (majority, TF-IDF+LR, zero-shot, fine-tuned DistilBERT, all by macro-F1) using Recharts.
- Fusion card: the ablation table and the sensitivity result (Section 15.3).
- Dataset card: names, links, licences and split sizes.

**System Status**: per model: loaded yes/no, checkpoint path, trained_at, device, headline metric; `incidents.json` generated_at and count; weather configured yes/no. No percentages that are not measured.

**Vision (YOLOv8n)**: upload an image and pick a source (officer_photo / drone / cctv / citizen_upload); show the annotated image, a detections table and the vision score. Remove the old "model pending" text.

**Urgency NLP (DistilBERT)**: a text box and a source picker; show the urgency badge, a 3-bar probability chart and the NLP score. Remove the flood-type badge and sentiment.

**Incident Analyzer (Fusion)**: optional image upload + optional text + the two source pickers, then one call to `/api/incident/analyze`. Results: RiskBadge, score, EvidenceMeter, conflict/single-source, the action, and `FusionBreakdown`. **Remove the weather input from the form.** Add a "Load test case" dropdown that fills in T1–T7 so the fusion behaviour can be shown live in the viva.

**Weather**: unchanged, with the context-only label at the top.

**Landing page**: update the copy to flood-only and list the two fine-tuned models and the simulated-data policy honestly.

### 11.5 Types
Add `Incident`, `VisionResult`, `NlpResult`, `FusionResult`, `EvaluationPayload` and `DataOrigin` to `services/api.ts`, matching Section 9.4 exactly. `npm run build` (type check) must pass.

---

## 12. Git workflow

- Branch `rebuild` off `2b7c2a4` ("PHASE 1 AND 2 COMPLETED"). Never commit to `main` directly; never force-push; never rewrite history.
- One commit per module, message prefix `M0:` … `M6:`, with a body listing what changed and the done-when evidence.
- `.gitignore`: `datasets/`, `training/runs/`, `models/nlp/*.safetensors`, `models/nlp/*.bin`, `backend/uploads/*` (keep `.gitkeep`), `__pycache__/`, `.venv/`, `node_modules/`, `.env`.
- Commit: `models/vision/best.pt`, both `metrics.json` files, `models/*/figures/`, `backend/data/incidents.json`, `backend/static/incidents/`.
- Tell the teammate about the branch before merging.

---

## 13. Honesty checklist (all must be true at M6)

- [ ] No response from any endpoint lacks `data_origin`.
- [ ] No page shows simulated or static data without a badge.
- [ ] `grep -rniE "earthquake|cyclone|wildfire|landslide|drought|tsunami|industrial|outbreak|segformer|is_fallback|zero-shot|sentiment" backend frontend/src` returns nothing, except `training/eval_urgency.py` baseline code, which is outside those folders.
- [ ] The System Status page shows no hardcoded health percentages.
- [ ] The weather value is never used in `fusion_engine.py` (enforced by a test that asserts the fusion signature has no weather parameter).
- [ ] The README says what is fine-tuned, on what data, with which metrics, and what is simulated.

---

## 14. Runbook (one day)

### 14.0 Pre-flight (you, ~45 min, before starting Claude Code)
1. `git status`, `git log --oneline -3`, then `git checkout -b rebuild`.
2. Download the vision dataset → `datasets/raw/flood_yolo/` (Section 5.1). **Look at about 20 images first.**
3. Download HumAID set 1 → `datasets/raw/humaid/` and extract it there (Section 5.2).
4. Install CUDA PyTorch (Section 7.1) and `pip install -r training/requirements.txt` once that file exists (or install the listed packages directly).
5. Run `python check_setup.py`; every line should be `[OK]`.
6. Put `CLAUDE.md` in the repo root and this file at `docs/MDRIS_CONTEXT.md`. Commit both as `M-1: add project context`.

### 14.1 Modules, done-when checks and time

| Module | Work | Done when | Time |
|---|---|---|---|
| **M0** Scope strip | Delete `mdris-mvp/`; remove non-flood strings, the COCO heuristic, SegFormer placeholders, zero-shot + sentiment from the backend; add `config.py`; update `.gitignore` | The Section 13 grep is clean; the backend starts (models may be missing → 503 is fine) | 45 min |
| **M1** Data | `common.py`, `prepare_vision.py`, `prepare_text.py`, `DATA.md` | Processed folders exist; stats are in `DATA.md`; **human spot-check** of 30 images with boxes drawn and 50 random mapped tweets | 45 min + 15 min for you |
| **M2** Vision | `train_yolo.py` (background), `eval_vision.py`, rewire `ai/vision/*` | `/api/vision/predict` returns real detections from `models/vision/best.pt`; `metrics.json` + figures saved | 90 min (mostly waiting) |
| **M3** NLP | `train_urgency.py`, `eval_urgency.py` (with baselines), rewire `ai/nlp/*` | `/api/nlp/analyze` returns fine-tuned urgency; `metrics.json` includes the baselines | 45 min |
| **M4** Fusion | `constants.py`, `fusion_engine.py`, `actions.py`, `/api/incident/analyze`, `tests/test_fusion.py` | `pytest` passes all of T1–T8 | 60 min |
| **M5** Honesty + demo | `generate_incidents.py`, `fusion_analysis.py`, new routers, `/api/evaluation`, real system status, all frontend changes from Section 11 | App runs end-to-end; every page badge-correct; `npm run build` passes | 2.5 h |
| **M6** Integration | Manual click-through of every page, `test_api_smoke.py`, README rewrite, `docs/EVALUATION.md`, the honesty checklist | Every Section 13 box ticked | 60 min |

**Parallelism inside a single agent:** while YOLO trains in the background (M2), Claude Code may write the M4 fusion code and tests, since they need no GPU. It must not start the DistilBERT training until YOLO has finished.

**Human checkpoints (the agent stops and waits):**
1. After M1: you review the spot-check images and tweets.
2. After M2 and M3: you look at the metrics. Red flags: mAP50 < 0.30, or NLP macro-F1 < 0.55 or below the TF-IDF baseline. Either one means you check the data before tuning.
3. Before merging `rebuild` into `main`: your teammate agrees.

### 14.2 Prompts for Claude Code (one per session or module)

Use one prompt per module rather than one giant prompt; it recovers better from errors.

**Kick-off (every session):**
```
Read CLAUDE.md and docs/MDRIS_CONTEXT.md fully before doing anything.
Run `git status` and `git log --oneline -5` and tell me which module was last completed.
```

**M0:**
```
Execute module M0 exactly as specified in docs/MDRIS_CONTEXT.md sections 9.2, 12, 13 and 14.1.
Do not touch training or datasets. Finish with the section 13 grep output and a backend startup log,
then commit as "M0: ...". Stop after committing.
```

**M1:**
```
Execute M1 per sections 5, 6 and 7.1. First inspect datasets/raw/ and print what you find; do not assume layouts.
If any raw vision class is not in the 5.1.1 remap table, or fewer than 1,500 text rows remain, stop and report.
Write DATA.md. Generate a spot-check folder: 30 random processed training images with boxes drawn, and
spot_check_text.csv with 50 random rows (text, humaid_label, urgency). Commit as "M1: ...", then stop for my review.
```

**M2:**
```
Execute M2 per sections 7.2, 7.4 and 9. Launch YOLO training as a background process logging to
training/runs/yolo_train.log and poll it. While it trains, you may implement M4 code and tests (section 8) but do not commit them yet.
When training finishes: run eval_vision.py, write models/vision/metrics.json and figures, rewire ai/vision to the new checkpoint,
verify /api/vision/predict on 3 test images, commit M2. Report mAP50 and per-class AP50, then stop.
```

**M3:**
```
Execute M3 per sections 7.3, 7.4 and 15.2. Confirm the GPU is free first. Train, then evaluate on the test split with all three
baselines, write models/nlp/metrics.json and figures, rewire ai/nlp, verify /api/nlp/analyze on 5 test rows,
commit M3. Report macro-F1 against the baselines, then stop.
```

**M4:**
```
Complete M4 per section 8 exactly (constants, formulas, action map, /api/incident/analyze). Make pytest pass T1–T8 with the
expected values in 8.4. Commit M4.
```

**M5:**
```
Execute M5 per sections 9.3, 10, 11 and 15.3. Build generate_incidents.py using the backend predict functions (no
reimplementation), then fusion_analysis.py, then the new/changed routers, then every frontend change in section 11.
Run the backend and `npm run build`. Commit M5.
```

**M6:**
```
Execute M6: write backend/tests/test_api_smoke.py, go through the section 13 checklist and show evidence for each item,
write docs/EVALUATION.md from the metrics files, and rewrite README.md (what is fine-tuned, on which data, metrics,
what is simulated, how to run). Commit M6 and summarise remaining limitations.
```

---

## 15. Evaluation plan

### 15.1 Vision (test split)
- mAP50, mAP50-95, and per-class precision, recall and AP50 (`model.val(split="test")`).
- Confusion matrix and PR curve.
- **6 failure cases:** match predictions to ground truth at IoU 0.5 and conf 0.25, rank images by (FN + FP), and save the worst 6 as side-by-side ground truth vs prediction in `models/vision/figures/failures/`.
- Optional baseline: COCO-pretrained YOLOv8n on the `person` and `vehicle` classes only, to show what fine-tuning adds for the flood-specific classes it cannot detect at all.

### 15.2 NLP (test split)
- Accuracy, **macro-F1** (headline), weighted-F1, per-class P/R/F1, and the confusion matrix.
- Per-event macro-F1, especially Kerala.
- **Baselines**, all on the same test split:
  1. Majority class.
  2. TF-IDF (1–2 grams, min_df=2) + LogisticRegression(class_weight="balanced", max_iter=2000).
  3. Zero-shot NLI: the zero-shot model the old backend used (read its name from the old `ai/nlp/predict.py` before deleting it; otherwise `typeform/distilbert-base-uncased-mnli`), with hypotheses:
     High → "someone urgently needs rescue, help or medical aid"; Medium → "damage, warnings or ongoing relief work"; Low → "sympathy, prayers or unrelated information".
- Results table: model · accuracy · macro-F1 · per-class F1.

### 15.3 Fusion (no ground truth, so evaluate behaviour)
1. **Unit tests** T1–T8 (Section 8.4).
2. **Ablation** over the 40 incidents, comparing against the full rule:
   (a) flat mean of S_v and S_n, (b) no reliability (all 1.0), (c) no renormalization (missing = 0), (d) weather at 30% (from the old design, with a constant city rainfall value).
   For each, report how many incidents change risk band and the Kendall τ between rankings.
3. **Sensitivity:** vary `VISION_WEIGHT` over {0.4, 0.5, 0.6, 0.7, 0.8} and scale all reliabilities by ±20%. Report the Kendall τ of the top-10 ranking against the default. High τ means the ranking is not fragile to the chosen constants.
4. Write everything to `models/fusion_analysis.json` and `docs/EVALUATION.md`; show it on the Model Evaluation page.

---

## 16. Viva defence notes

1. **Why flood only?** Depth: two fine-tuned models and a documented fusion rule in one semester.
2. **Why YOLO and not segmentation?** Commanders act on objects (a person in water, a trapped vehicle), not pixel masks. Detection classes feed the fusion score and action map directly.
3. **Why DistilBERT?** It is small enough to fine-tune on a 4 GB GPU in minutes and fast on a CPU at inference, and it beats the TF-IDF and zero-shot baselines (show the chart).
4. **Your urgency labels are not real urgency labels.** Correct, and this is stated. We mapped HumAID's information-type categories to urgency with written reasons (5.2.3). The mapping is the weakest link, and we evaluate against it honestly.
5. **Why is weather excluded?** A city rainfall figure is the same for every incident in the city, so it cannot rank incidents; it only shifts them all. Ablation (d) in Section 15.3 shows what adding it would do to the bands.
6. **Why not learned fusion?** There are no labelled fusion targets. A documented rule is auditable, and the sensitivity analysis shows the ranking is stable.
7. **What exactly is your contribution?** Confidence × source-reliability weighting, renormalization over available evidence, a separate evidence strength value, and a conflict flag, all visible in the breakdown panel.
8. **Does a citizen photo always lower the score?** No. Reliability lowers that source's influence and the evidence strength; the score moves toward the other source (tests T5a/T5b).
9. **What is simulated?** Incident locations, times and the pairing of images with texts. The images, texts and all model outputs are real. Say this first.
10. **Limitations to volunteer:** domain shift (non-Indian imagery, English tweets vs Tamil and Tanglish reports), the urgency mapping, uncalibrated weights, no ground-truth priority, a small vision dataset, and the risk of pool images in the "drowning" class.

---

## 17. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| CPU-only PyTorch installed by accident | High | Scripts assert CUDA and exit |
| Out-of-memory on 4 GB | Medium | batch 8 → 4; never train both models together |
| Windows dataloader hang | Medium | `workers=0`; `__main__` guard |
| Unknown class names in the export | Medium | Remap script stops; no guessing |
| Kerala missing from HumAID set 1 | Medium | Dataverse full set; the script works with any flood events |
| Retweet leakage inflating NLP scores | High if ignored | Deduplicate before splitting |
| Medium class dominates | High | Class-weighted loss; macro-F1 headline |
| Low mAP on rare classes | Medium | Report honestly; show failure cases; do not over-tune |
| The agent improvises fake fallbacks | Medium | `CLAUDE.md` hard rules + honesty checklist |
| Scope creep (DB, auth, Docker) | High | Out of scope; `CLAUDE.md` forbids it |
| Teammate conflict | Medium | Work on `rebuild`; merge only after agreement |

---

## 18. Quick reference

```
Vision   YOLOv8n fine-tuned · 4 classes: person_in_water, person_in_vehicle, person, vehicle
         data: Roboflow "detecting humans in floods" (GNOSEF) · imgsz 640 · batch 8 · 60 ep
NLP      DistilBERT fine-tuned · 3-level urgency Low/Medium/High
         data: HumAID flood events (~11k before dedupe) · max_len 128 · bs 16 · lr 2e-5 · 4 ep · weighted CE
Fusion   S_v noisy-OR, S_n expected urgency · eff = W_mod × reliability × conf
         priority = 100·Σ(eff·S)/Σeff · evidence = Σeff · conflict if |S_v−S_n|>0.5
         weights 0.6/0.4 · bands 75/55/35
Weather  display only
Demo     40 simulated incidents, real models, Chennai localities, labelled SIMULATED
GPU      RTX 2050 4 GB · CUDA torch mandatory · train models sequentially
Git      branch rebuild off 2b7c2a4 · one commit per module M0–M6
Run      backend: cd backend && uvicorn main:app --reload   frontend: cd frontend && npm run dev
```
