# MDRIS overnight run — morning report

Branch: **`rebuild/flood-scope`** (`rebuild` could not be created, see §4). `main` untouched, nothing pushed.

## 1. Status

| Module | Status | Commit |
|---|---|---|
| M-1 context docs | DONE | `e1e27b6` |
| M0 scope strip | DONE | `874b859` |
| M1 data | DONE (gate PASS). **Vision data NEEDS REVIEW:** the export has one class only | `20bfec2` |
| M2 vision | DONE (gate PASS: test mAP50 0.384 ≥ 0.30) | `ff44fd3` |
| M3 NLP | DONE (gate PASS: macro-F1 0.771 ≥ 0.55 and ≥ TF-IDF 0.749) | `c5539b1` |
| M4 fusion | DONE (T1–T8 green) | `fff943d` |
| M5 honesty + demo | DONE | `0497f17` |
| M6 integration | DONE | see `git log -1` (`M6:`) |

## 2. Headline results (all on held-out test splits)

**Vision, fine-tuned YOLOv8n:** mAP50 **0.384**, mAP50-95 0.181. Per-class AP50: `person` 0.384
(P 0.629, R 0.330, 1,329 instances). `person_in_water`, `person_in_vehicle` and `vehicle` have **no data**, so they are
not evaluable (null). The COCO YOLOv8n baseline gets person AP50 0.137 against 0.365 for the fine-tuned model, both
scored with the same code. Validation mAP50 reached 0.76, but that number is inflated by Roboflow's train/valid overlap.

**NLP, fine-tuned DistilBERT (Low/Medium/High):**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority | 0.566 | 0.241 |
| Zero-shot NLI (`typeform/distilbert-base-uncased-mnli`) | 0.303 | 0.251 |
| TF-IDF + LR | 0.786 | 0.749 |
| **Fine-tuned DistilBERT** | **0.805** | **0.771** |

- Per-class F1: Low 0.779, Medium 0.849, High 0.684.
- Per-event macro-F1: Kerala 0.762, Maryland 0.890, Midwest 0.625, Sri Lanka 0.724.
- The gain over TF-IDF is small (+0.02).

**Fusion:** 12/12 tests pass. That covers T1–T8 with the exact §8.4 values, a no-weather-input check, a
detection-threshold check and an unknown-source check.

The 40 demo incidents break down as Critical 3, High 4, Moderate 22, Low 11, with 1 conflict.

Ablation against the full rule:

| Variant | Band changes | Kendall τ |
|---|---|---|
| (a) Flat mean | 1 | 0.92 |
| (b) No reliability | 0 | 0.96 |
| (c) No renormalization | 9 | 0.69 |
| (d) Weather at 30% | 6 | 1.00 (every score shifts by +7) |

Sensitivity: the top-10 ranking keeps τ between 0.82 and 1.00 for VISION_WEIGHT 0.4–0.8. Scaling all reliabilities
changes nothing unless the 1.0 cap applies. Full tables are in `docs/EVALUATION.md` and on the Model Evaluation page.

## 3. Human review needed (do these first)

1. **The vision dataset is the wrong version.** `datasets/raw/flood_yolo/` is GNOSEF **v2**, which labels only
   `person`. The spec expected `car / drowning / person / in-car`.
   - Consequence: the detector can never say "person in water" or "vehicle".
   - Vision scores stay low (at most 0.45 per detection), so few incidents reach High or Critical from images.
   - **Fix:** on Roboflow, pick a version of the GNOSEF project that has 4 classes (or use the t5 fallback), export it
     as YOLOv8 into `datasets/raw/flood_yolo/`, then run:
     ```
     .venv-gpu\Scripts\python training\prepare_vision.py
     .venv-gpu\Scripts\python -u training\train_yolo.py > training\runs\yolo_train.log 2>&1
     .venv-gpu\Scripts\python training\eval_vision.py
     .venv-gpu\Scripts\python training\generate_incidents.py
     .venv-gpu\Scripts\python training\fusion_analysis.py
     .venv-gpu\Scripts\python training\write_evaluation.py
     ```
2. **Vision spot-check:** look at `datasets/processed/spot_check/vision/` (30 images with boxes drawn).
   - Are the boxes right? Are there any swimming-pool images?
   - In the 4 I opened: real flood video frames, no pools. Some people were left unboxed, and several frames are
     heavily rotated.
3. **Text spot-check:** open `datasets/processed/spot_check/text.csv` (50 rows). Does the urgency mapping look
   sensible? Donation drives landing in Medium is the known weak point.
4. **Tweet confidentiality.** HumAID's terms ask that tweet *contents* be kept confidential.
   - As the spec requires, `backend/data/incidents.json` contains 34 tweet texts.
   - Decide before pushing this branch to a public GitHub remote.
   - Evidence files contain labels only.
5. **Branch name.** Run `git branch -m rebuild/flood-scope rebuild` if you want the spec's name. Tell your teammate
   before merging.

## 4. Decisions made overnight

These are summarised here; the full reasoning is in `docs/PROGRESS.md` (decisions 1–21).

- **Branch:** the `rebuild` ref conflicts with the existing `rebuild/flood-scope`, so all work is on
  `rebuild/flood-scope`.
- **Environment:** the only CUDA torch is under Python 3.13 (`torch 2.7.1+cu118`); 3.11 has only CPU torch.
  - Created `.venv-gpu` with `--system-site-packages`, so torch was never reinstalled.
  - ultralytics 8.2.60 is installed with `--no-deps` and needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for our own
    checkpoints.
  - transformers is **4.46.3** instead of 4.42.4, because 4.42.4 has no Python 3.13 wheels.
- **HumAID:** set 1 had only about 1,300 flood rows, so set 2 (Kerala, Midwest) was also downloaded from the same
  source. HumAID had already de-duplicated the tweets: my dedupe step removed 0 rows.
- **Vision test split:** carved from train, 10% of source stems, so augmented copies of one image never straddle
  train and test.
- **Failure cases:** the worst 6 *distinct* source images. Without this, all six were copies of one frame.
- **Code written ahead of its module:** the fusion engine, the incident endpoint and the four AI pages went in with
  M0. M2 accidentally also carried two staged M5 file deletions. History was not rewritten.
- **Frontend build fix:** the build was already broken at HEAD. Added `@tailwindcss/postcss` so Tailwind 4 works.
- **Honesty fixes:**
  - The hardcoded "All Systems Nominal" is replaced by the live model count.
  - A failed weather request now returns 502, and a missing key returns `{"configured": false}`.
  - Default sources are the unverified ones.
- **Incident Analyzer "Load test case":** it sends T1–T7 as detections and probabilities to `/api/fusion/analyze` and
  shows the expected values next to the result.
- **§13 grep:** the only match is inside one real tweet in `incidents.json`, which mentions the 2004 tsunami. It was
  kept, not filtered out.
- **`backend/.env`:** removed from git tracking. Its weather key was empty.

## 5. Known issues and limitations

- The single-class vision data (above). Validation metrics are inflated by the train/valid overlap.
- Urgency labels come from HumAID *information types*, not urgency judgements.
  - Medium is 57% of the data.
  - The Midwest event scores lowest (0.625).
- Domain shift: the images are mostly non-Indian video frames and the text is English only.
- Fusion constants are documented choices, not calibrated. There is no ground-truth priority.
- The weather key is empty, so the weather card shows "Weather API key not configured".
- Inference runs on CUDA when it is available (`DEVICE=auto`). Set `DEVICE=cpu` in `backend/.env` if you need to
  keep the GPU free.

## 6. How to see it

```powershell
# Terminal 1: backend
cd backend
..\.venv-gpu\Scripts\python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: frontend
cd frontend
npm run dev          # open http://localhost:5173

# Tests
cd backend; ..\.venv-gpu\Scripts\python -m pytest -q      # 35 passed
cd frontend; npm run build                                 # type check
```

5-minute demo, in this order:

1. **Landing page:** what MDRIS is and the simulated-data policy.
2. **Command Dashboard:** the SIMULATED banner, KPIs, and the priority queue. Click the top row to open the drawer,
   which shows the annotated image, the tweet, the urgency bars and the fusion breakdown.
3. **Incident Map:** band colours, faded markers for weak evidence, and the "conflicts only" filter.
4. **Incident Analyzer:** run T1, then T4 (conflict), then T5a vs T5b (reliability lowers influence, not the score
   blindly). Then upload an image and enter text for a live analysis.
5. **Model Evaluation:** vision metrics and failure cases, the NLP baseline chart, the fusion ablation table.
6. **System Status:** real load state, checkpoints and metrics.

## 7. Time log (29 Sep 2026, IST)

| Step | Start | End |
|---|---|---|
| Setup, environment, HumAID download | 20:10 | 20:45 |
| M0 scope strip | 20:25 | 21:05 |
| M1 data | 20:40 | 21:15 |
| YOLO training (60 epochs) | 20:53 | 22:42 (109.4 min) |
| M4 / M5 code written during YOLO training | 21:05 | 21:30 |
| Shell commands blocked by a failing permission check (training unaffected) | ~22:20 | ~22:38 |
| M2 eval + commit | 22:42 | 22:48 |
| DistilBERT training | 22:46 | 22:50 (3.7 min) |
| M3 eval + commit | 22:50 | 22:53 |
| M4 commit, M5 generation + commit | 22:53 | 22:57 |
| M6 | 22:57 | ~23:05 |
