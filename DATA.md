# MDRIS datasets

Both datasets are **git-ignored** (`datasets/`). This file records where they came from, how they were processed and
what is known to be wrong with them. Numbers below were printed by `training/prepare_vision.py` and
`training/prepare_text.py` (seed 42); the machine-readable versions are `datasets/processed/*/stats.json`.

---

## 1. Vision: "detecting humans in floods" (GNOSEF), version 2

| | |
|---|---|
| Source | https://universe.roboflow.com/gnosef/detecting-humans-in-floods-gukif (version 2, exported 2025-03-17, YOLOv8 format) |
| Licence | **Public Domain**, as stated in the export's `README.roboflow.txt` and `data.yaml` (`license: Public Domain`). The Roboflow page shows this as CC0. |
| Raw layout | `datasets/raw/flood_yolo/{train,valid}/{images,labels}`, `data.yaml`; **no `test/`** |
| Raw size | 5,046 images (4,539 train + 507 valid). Roboflow pre-processing: auto-orient, resize to 840×640 (stretch). Augmentation: 3 versions of each *training* source image (flip, brightness ±20%, blur ≤2.5 px, 1.33% salt-and-pepper noise). |
| Raw classes | **one class: `person`** |

### 1.1 Canonical remap (spec §5.1.1)

| Raw class | Canonical class | ID | Instances (train / val / test) |
|---|---|---|---|
| — | `person_in_water` | 0 | 0 / 0 / 0 |
| — | `person_in_vehicle` | 1 | 0 / 0 / 0 |
| `person` | `person` | 2 | 9,309 / 1,097 / 1,329 |
| — | `vehicle` | 3 | 0 / 0 / 0 |

**Classes with fewer than 100 training instances: `person_in_water`, `person_in_vehicle`, `vehicle` (all zero).**
This export is the single-class version of the dataset. The spec expected the 4-class version
(`car`, `drowning`, `person`, `in-car`). The processed dataset keeps all four canonical names so the pipeline is
unchanged, but a model trained on it can only detect `person`. See the morning report for how to fix this.

### 1.2 Splits

| Split | Images | Of which no objects | Source |
|---|---|---|---|
| train | 4,152 | 783 | raw `train/` minus the carved test set |
| val | 507 | 98 | raw `valid/`, unchanged |
| test | 387 | 39 | **carved from raw `train/`** (no `test/` in the export) |

- **How test was carved:** 10% of training *source stems* (60 of 597), seed 42. A stem is the file name before
  `_jpg.rf.<hash>`; all augmented copies of a stem go to the same split, so no augmented copy of a test image is in
  train. Stems are reused by different source images (e.g. `yt-false-0001` has 81 files = 27 sources × 3), so this
  grouping is conservative.
- **Known leakage in Roboflow's own split:** 297 stems occur in both raw `train/` and raw `valid/`. A flip-invariant
  difference-hash check (256-bit, `datasets/processed/flood_yolo/near_duplicates.json`) finds:

  | Split vs train | Images | min Hamming ≤ 10 | ≤ 20 | ≤ 40 | median min distance |
  |---|---|---|---|---|---|
  | val | 507 | 11 | 62 | 199 | 51 |
  | test (carved) | 387 | 0 | 12 | 90 | 65 |

  Validation mAP (used only for early stopping and best-epoch selection) is therefore optimistic. **All reported
  vision metrics are on the carved test split.**
- Every image has a label file; all class ids are 0–3 and all coordinates are in [0, 1] (the script stops otherwise).

### 1.3 Known biases and limitations

- Single class: no distinction between a person standing on a street, wading, or drowning.
- Mostly low-resolution video frames (YouTube, news clips, many with watermarks), several heavily rotated; non-Indian
  settings dominate. Domain shift to Chennai CCTV/drone/officer photos is expected.
- Label recall is imperfect: in the spot-check some visible people are not boxed.
- Spot-check images: `datasets/processed/spot_check/vision/` (30 random processed training images, boxes drawn).

---

## 2. Text: HumAID, flood events

| | |
|---|---|
| Source | HumAID (Alam, Qazi, Imran, Ofli; ICWSM 2021), CrisisNLP / QCRI, https://crisisnlp.qcri.org/humaid_dataset |
| Files | `HumAID_data_events_set1_47K.tar.gz` and `HumAID_data_events_set2_29K.tar.gz` from `https://crisisnlp.qcri.org/data/humaid/`, extracted into `datasets/raw/humaid/events_set1/` and `events_set2/` |
| Licence | CC BY-NC 4.0 (non-commercial). The bundled `Licensing.txt` also asks users to use the data only for humanitarian-computing research, keep tweet contents confidential (tweet ids may be shared), and cite the paper. |
| Citation | Firoj Alam, Umair Qazi, Muhammad Imran, Ferda Ofli. *HumAID: Human-Annotated Disaster Incidents Data from Twitter with Deep Learning Benchmarks.* ICWSM 2021. |

Set 1 contains only two flood events (Maryland, Sri Lanka; ~1,300 rows, below the 1,500 minimum), so set 2 was also
downloaded, as §5.2 prescribes when Kerala is missing.

### 2.1 Events used (every folder whose name contains `flood`)

| Event | Raw rows | Train | Val | Test |
|---|---|---|---|---|
| `kerala_floods_2018` (India) | 7,984 | 5,598 | 1,198 | 1,188 |
| `midwestern_us_floods_2019` | 1,880 | 1,304 | 271 | 305 |
| `maryland_floods_2018` | 742 | 530 | 115 | 97 |
| `srilanka_floods_2017` | 560 | 384 | 91 | 85 |
| **Total** | **11,166** | **7,816** | **1,675** | **1,675** |

(`maryland_floods_2018` is stored inside `hurricane_matthew_2016/` in the archive; the script selects by the name of
the folder that holds the `.tsv`.)

### 2.2 Cleaning (spec §5.2.1, in this order)

1. Load all train/dev/test files of the four events (columns `tweet_id`, `tweet_text`, `class_label` + `event`).
2. Drop `dont_know_cant_judge` and empty texts: **0 rows** dropped (none present).
3. Normalize: strip a leading `RT @x:`, URLs → `URL`, `@mentions` → `@USER`, collapse whitespace.
   HumAID had already removed most URLs (only truncated `http` fragments remain) and retweets.
4. Deduplicate on normalized text before splitting: **0 duplicates** removed. HumAID was already de-duplicated by
   its authors; the check is kept so the pipeline is safe on other data.
5. Map to urgency (§5.2.3).
6. Stratified 70/15/15 split on urgency, seed 42. HumAID's own splits are **not** used: labels were remapped and the
   data re-deduplicated, so the official splits no longer apply cleanly.
7. No text appears in more than one split (checked).

### 2.3 Urgency mapping and distribution

| Urgency | HumAID categories (rows) | Total | Train / Val / Test |
|---|---|---|---|
| **High** | requests_or_urgent_needs (670), injured_or_dead_people (465), missing_or_found_people (140), displaced_people_and_evacuations (56) | 1,331 (11.9%) | 932 / 199 / 200 |
| **Medium** | rescue_volunteering_or_donation_effort (5,474), infrastructure_and_utility_damage (532), caution_and_advice (316) | 6,322 (56.6%) | 4,425 / 949 / 948 |
| **Low** | other_relevant_information (1,421), sympathy_and_support (1,150), not_humanitarian (942) | 3,513 (31.5%) | 2,459 / 527 / 527 |

### 2.4 Known biases and limitations

- **The labels are information types, not urgency.** The mapping is a documented project decision.
  `rescue_volunteering_or_donation_effort` alone is 49% of all rows and mixes live rescues with donation drives, so
  Medium dominates. Training uses class-weighted loss and macro-F1 is the headline metric.
- 72% of rows come from one event (Kerala 2018): per-event scores for the three small events are noisy.
- English tweets only; real Tamil Nadu reports would include Tamil and Tanglish SMS and call transcripts.
- Spot-check rows: `datasets/processed/spot_check/text.csv` (50 random rows: text, humaid_label, urgency).

---

## 3. Reproduce

```
python training/prepare_vision.py   # datasets/raw/flood_yolo -> datasets/processed/flood_yolo
python training/prepare_text.py     # datasets/raw/humaid     -> datasets/processed/flood_text
```
