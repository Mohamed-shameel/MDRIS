# MDRIS rebuild — progress log

Overnight autonomous run started 2026-09-29 20:10 IST. Spec: `docs/MDRIS_CONTEXT.md`. Rules: `CLAUDE.md`.

## Facts recorded before deleting old code

- **Old zero-shot model (for the M3 baseline):** `typeform/distilbert-base-uncased-mnli`
  (from the old `backend/ai/nlp/predict.py`, `pipeline("zero-shot-classification", model=...)`).
- Old sentiment model (removed, not used anywhere): `distilbert-base-uncased-finetuned-sst-2-english`.

## Environment

- GPU: NVIDIA GeForce RTX 2050, 4 GB.
- Python installs found: 3.10 (no torch), 3.11 (`torch 2.3.1+cpu`, ultralytics 8.2.60, transformers 4.42.4),
  3.13 (`torch 2.7.1+cu118`, CUDA available). The repo `.venv` (3.13) was empty.
- **Training/backend environment used overnight:** `.venv-gpu/` = Python 3.13 venv created with
  `--system-site-packages`, so it reuses the existing CUDA torch 2.7.1+cu118 unchanged (torch was not reinstalled).
  See "Decisions made overnight" for the version pins this forced.

## Decisions made overnight

1. **Branch.** `git checkout -b rebuild` fails: the ref `rebuild/flood-scope` already exists and Git cannot hold both
   `rebuild` and `rebuild/...`. Work continues on the existing `rebuild/flood-scope` branch (not `main`), which already
   contains the teammate's "Scope correction" commit on top of 2b7c2a4. Rename in the morning if wanted:
   `git branch -m rebuild/flood-scope rebuild`.
2. **Context doc location.** `MDRIS_CONTEXT.md` was at the repo root; moved to `docs/MDRIS_CONTEXT.md` as the spec says,
   committed with `CLAUDE.md` as `M-1: add project context`.
3. **Python 3.13 instead of 3.10/3.11.** The only CUDA-enabled torch on this machine is in Python 3.13. Installing CUDA
   torch into 3.11 would change a torch build (forbidden). On 3.13:
   - `ultralytics==8.2.60` (spec pin) is kept, installed with `--no-deps` because it pins `numpy<2`, which has no 3.13
     wheel. It runs on numpy 2.3.3. torch ≥ 2.6 defaults `torch.load(weights_only=True)`, which ultralytics 8.2.60 does
     not handle, so our code sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` before importing ultralytics. It only ever loads
     our own checkpoints (`backend/yolov8n.pt`, `models/vision/best.pt`).
   - `transformers==4.42.4` cannot be installed on 3.13 (its `tokenizers<0.20` has no 3.13 wheel and `numpy<2` pin).
     **Used `transformers==4.46.3`** (closest release with 3.13 wheels) and `accelerate<1.2`.
   - `backend/requirements.txt`: `pydantic` pin relaxed to `>=2.7.4,<3` (2.7.4 has no 3.13 wheel), torch removed from
     the file (install the CUDA or CPU build separately, as §7.1 says), transformers bumped to 4.46.3.
4. **HumAID download.** `datasets/raw/humaid/` was empty. Downloaded set 1 (as instructed); it contains only Maryland
   and Sri Lanka floods (~1,300 rows, below the 1,500 floor). §5.2 says to use the full set if Kerala is missing, so
   set 2 (`HumAID_data_events_set2_29K.tar.gz`, same CrisisNLP site and licence) was also downloaded; it has
   `kerala_floods_2018` and `midwestern_us_floods_2019`. Nothing under `datasets/raw/` was modified or deleted.
5. **Frontend build was already broken at HEAD** (Tailwind 4 needs the `@tailwindcss/postcss` plugin; `@tailwind`
   directives no longer work; plus unused-import TS errors). Installed `@tailwindcss/postcss@4.3.3` as a dev dependency,
   switched `index.css` to `@import "tailwindcss"` + `@config` + a class-based `dark` variant (keeps the existing
   dark-mode toggle working).
6. **`backend/.env` was tracked by git.** Removed from the index with `git rm --cached` (file kept on disk, secret
   values preserved) and git-ignored. Its `OPENWEATHER_KEY` was empty, so nothing secret is in history.
   Tracked `__pycache__/*.pyc` files were also removed from the index.
7. **Code written earlier than its module.** The old fusion engine (weather at 30%, `is_fallback`) had to go in M0, and
   the vision/NLP scores are defined by the fusion formulas, so the new `fusion_engine.py`, `actions.py`,
   `schemas.py`, `constants.py`, the model registry, the vision/NLP inference code and `/api/incident/analyze` were
   written in M0. Without trained models they return 503. M2/M3 add the models and evidence; M4 adds the T1–T8 tests.
   The four AI-module frontend pages (Vision, NLP, Incident Analyzer, Weather), the typed API client and the shared
   components were also rewritten in M0 instead of being patched twice.
8. **Default sources.** When a caller does not say where an input came from, the backend assumes the unverified sources
   (`citizen_upload`, `citizen_sms`). The UI always sends an explicit source.
9. **Weather errors.** A failed weather request now returns HTTP 502 with the reason (it used to return zeros with
   `is_fallback`). Without a key: `{"configured": false}`.
10. **"Load test case" on the Incident Analyzer.** T1–T7 are defined as detections + probabilities, not images/texts,
    so the dropdown sends them to `/api/fusion/analyze` and shows the expected values from §8.4 next to the result.
11. **Vision dataset is single-class.** The export in `datasets/raw/flood_yolo/` is GNOSEF **version 2**, which has one
    raw class, `person` (5,046 images incl. 3x augmentation). `person` is in the remap table, so this is not a hard
    stop, but `person_in_water`, `person_in_vehicle` and `vehicle` have **zero** instances: the fine-tuned detector can
    only ever output `person`, so S_v is capped by w=0.45 per detection. Continued with the 4-class pipeline unchanged
    (it will work as-is on a 4-class export). **Human action:** download a GNOSEF version with `car/drowning/person/in-car`
    (or the t5 fallback) and re-run M1–M2 (commands in the morning report).
12. **Test split carved by source image.** The export has no `test/`. Roboflow file names are `<stem>_jpg.rf.<hash>.jpg`
    and each training source has 3 augmented copies, so 10% of *stems* (60 of 597, seed 42) went to test (387 files).
    Stems are reused across different source images (e.g. many `yt-false-0001` frames), so grouping by stem is
    conservative.
13. **Roboflow's own `valid/` overlaps `train/`.** 297 of the valid stems also occur in train, and a flip-invariant
    difference-hash check finds 62/507 valid images within Hamming distance 20 (of 256) of a training image, versus
    12/387 for our carved test split. Validation metrics (used for early stopping and best-epoch selection) are
    therefore optimistic; **test metrics are the ones reported**. `valid/` was kept as-is (the spec does not allow
    re-splitting it). Details in `datasets/processed/flood_yolo/near_duplicates.json` and DATA.md.

14. **§13 grep vs. the zero-shot baseline label.** The spec requires the zero-shot baseline on the Model Evaluation
    page, but "zero-shot" is in the §13 grep. The page reads the baseline's display name from `metrics.json`
    (written by `training/eval_urgency.py`), so no such string is hardcoded in `backend/` or `frontend/src`.
15. **Sidebar "System Health: All Systems Nominal"** was a hardcoded claim; replaced by the live model load state from
    `/api/system-status`. The top-bar bell's permanent red "unread" dot was removed (the bell links to Alerts).
16. **Failure cases = worst 6 distinct source images.** Ranking all test files by FN+FP picked six augmented
    copies of the same crowd frame. `eval_vision.py` keeps the worst file per source stem instead.
17. **Vision class metrics without test instances are `null`, not 0** (`person_in_water`, `person_in_vehicle`,
    `vehicle`), so mAP50 = person AP50.
18. **M2 commit also contains two M5 deletions.** `backend/data/mock_data.py` and `backend/routers/analytics.py` had
    been staged with `git rm` before the M2 commit and went into it. History was not rewritten; M5 completes the
    replacement (`/api/evaluation`, static resources).
19. **NLP evidence files omit tweet text** (row index + labels only), because HumAID's terms ask for tweet contents to
    be kept confidential. The spec-mandated `incidents.json` does contain 34 texts: see the morning report.
20. **§13 grep hit inside real data.** `backend/data/incidents.json` contains one held-out Kerala-flood tweet that
    mentions the 2004 tsunami. It is dataset text, not code or scope; it was kept (filtering tweets to satisfy a
    grep would be data manipulation). The grep over code (`--exclude=incidents.json`) returns nothing.
21. **Ablation (c) and (d) definitions** (spec leaves them open): (c) a missing modality is counted as present with
    S = 0 and its full modality weight; (d) 0.7 × priority + 30 × W with the old rule-based weather score for one
    constant city reading (42 mm/h, 55 km/h, 90% → W = 0.65). Reliability scaling ×1.2 is capped at 1.0 (uniform
    scaling alone cancels in the ratio).

## Modules

| Module | Status | Commit | Evidence |
|---|---|---|---|
| M0 scope strip | DONE | (see git log, `M0:`) | §13 grep returns nothing (exit 1); backend starts, missing models logged at ERROR, `/api/vision/predict` and `/api/nlp/analyze` return 503 with the reason; `/api/fusion/analyze` T1 input returns 87.0 / Critical / 0.88 |
| M1 data | DONE (gate PASS; vision content NEEDS REVIEW) | (see git log, `M1:`) | Gate: vision train/val/test = 4,152/507/387 images, 100% have label files, 0 parse errors, all ids in 0–3, all coords in [0,1]; text 11,166 rows after dedupe (0 duplicates), all 3 urgency classes in every split, 0 texts shared between splits. Caveat: vision export is single-class (`person` only), see decision 11. Spot-check: `datasets/processed/spot_check/` |
| M2 vision | DONE (gate PASS) | (see git log, `M2:`) | 60 epochs, 109.4 min, batch 8, workers 2 (no OOM). **Test mAP50 0.384, mAP50-95 0.181**; person P 0.629 / R 0.330 / AP50 0.384 (1,329 instances); other 3 classes: no instances. COCO yolov8n baseline person AP50 0.137 vs fine-tuned 0.365 (same simple AP50 code). Val mAP50 was 0.76 (inflated by train/val overlap). `/api/vision/predict` on 3 test images: 200, real detections (0/1/2), saved in `docs/evidence/m2_vision_predict.json` |
| M3 NLP | DONE (gate PASS) | (see git log, `M3:`) | Early-stopped after epoch 3 (best val macro-F1 0.790 at epoch 2), 3.7 min, batch 16. **Test acc 0.805, macro-F1 0.771, weighted-F1 0.807**; F1 Low 0.779 / Medium 0.849 / High 0.684. Baselines (test macro-F1): majority 0.241, TF-IDF+LR 0.749, zero-shot NLI (typeform/distilbert-base-uncased-mnli) 0.251. Per-event macro-F1: Kerala 0.762, Maryland 0.890, Midwest 0.625, Sri Lanka 0.724. `/api/nlp/analyze` on 5 test rows: 200, 5/5 match the mapped label (`docs/evidence/m3_nlp_analyze.json`) |
| M4 fusion | DONE | (see git log, `M4:`) | `pytest -q tests/test_fusion.py`: 12 passed (T1–T8 with the exact §8.4 values, no-weather signature test, threshold and unknown-source tests). Engine, constants, action map, schemas and `/api/incident/analyze` were committed in M0 (decision 7); unchanged since, no expected value or constant was altered |
| M5 honesty + demo | DONE | (see git log, `M5:`) | 40 incidents (28 paired: 17 concordant + 11 random; 6 image-only; 6 text-only), bands Critical 3 / High 4 / Moderate 22 / Low 11, 1 conflict. Ablation (band changes, τ all): (a) flat mean 1, 0.923; (b) no reliability 0, 0.959; (c) no renormalization 9, 0.692; (d) weather 30% 6, 1.000. Sensitivity τ(top-10) 0.82–1.00 over VISION_WEIGHT 0.4–0.8. `npm run build` passes (0 type errors); uvicorn starts with both models loaded; all 12 GET endpoints return 200 with `data_origin`; `pytest -q` 35 passed |
| M6 integration | DONE | (see git log, `M6:`) | `backend/tests/test_api_smoke.py` hits every §9.3 endpoint (BLOCKED = empty; both models real). `pytest -q`: 35 passed. `npm run build`: passes. `docs/EVALUATION.md` generated from the JSON files; README rewritten |

## §13 honesty checklist (M6 evidence)

- [x] **No response lacks `data_origin`.** `test_api_smoke.py` asserts `data_origin` on every JSON endpoint of §9.3
  (35 passed). Live uvicorn check: all 12 GET endpoints returned 200 with `data_origin`
  (live / simulated / static_demo / model as specified).
- [x] **No page shows simulated or static data without a badge.** Dashboard, Map, Alerts: `SimulationBanner` +
  `DataOriginBadge "SIMULATED"`; IncidentDrawer: SIMULATED badge; Resources: STATIC DEMO badge + "Not connected to any
  resource system."; Vision/NLP/Incident Analyzer/Evaluation: MODEL OUTPUT; System Status/Weather: LIVE.
- [x] **§13 grep** over `backend frontend/src` (code): no matches (exit 1). One match exists inside
  `backend/data/incidents.json`, in a real held-out tweet (decision 20).
- [x] **System Status has no hardcoded percentages.** Page renders only `/api/system-status` fields (load state,
  checkpoint, device, trained_at, headline test metric, incidents file age, versions). Sidebar "All Systems Nominal"
  replaced by the live model count.
- [x] **Weather never used in fusion.** `fusion_engine.py` mentions weather only in its docstring;
  `test_fusion_has_no_weather_input` asserts `fuse()` and `FusionRequest` have no weather/rain field (passes).
- [x] **README** states what is fine-tuned, on what data, with which test metrics, and what is simulated.

## Next session must know

- Everything is committed on `rebuild/flood-scope`. Nothing is left running.
- To re-run with a 4-class vision export: replace `datasets/raw/flood_yolo/`, then
  `prepare_vision.py` → `train_yolo.py` → `eval_vision.py` → `generate_incidents.py` → `fusion_analysis.py` →
  `write_evaluation.py` (commands in docs/MORNING_REPORT.md).

## After the overnight run (2026-09-30)

- **Incident map tiles:** CARTO raster tiles now require an API key (every tile was an "API KEY REQUIRED" image);
  switched to tile.openstreetmap.org.
- **Frontend restyle at the user's request** (reference images in `D:\Downloads\proto 188\flow`): light
  blue-white console theme, navy text, Barlow Semi Condensed display / Inter body / JetBrains Mono fonts
  (self-hosted via @fontsource), light sidebar with section labels, pastel chips, left-accent banners,
  saffron/green stripe accents. This replaces §11.1's "dark mode by default"; the dark toggle was removed.
  The reference's national emblem and government branding were deliberately NOT copied (restricted use; a
  student project must not look like an official console). All honesty badges and risk colours are unchanged.
