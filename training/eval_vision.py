"""
Evaluate the fine-tuned YOLOv8n on the TEST split (docs/MDRIS_CONTEXT.md sections 7.4, 15.1)
=============================================================================================
1. ultralytics model.val(split="test"): mAP50, mAP50-95, per-class P / R / AP50,
   confusion matrix and PR curve (copied to models/vision/figures/).
2. Failure cases: predictions at conf 0.25 matched to ground truth at IoU 0.5
   (same class); images ranked by FN + FP; the worst 6 distinct source images saved side by side
   (ground truth | prediction) in models/vision/figures/failures/.
3. Optional baseline (section 15.1): COCO-pretrained yolov8n.pt with COCO
   `person` -> person and car/motorcycle/bus/truck -> vehicle. Both models are
   scored with the same simple AP50 code below, so the comparison is like for like.
4. Writes models/vision/metrics.json (section 7.4 schema).

Usage:  python training/eval_vision.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from common import (
    BASE_YOLO_WEIGHTS,
    CLASS_NAMES,
    COCO_TO_CANONICAL,
    PROCESSED_VISION_DIR,
    RUNS_DIR,
    SEED,
    VISION_DATASET,
    VISION_MODEL_DIR,
    allow_trusted_checkpoints,
    require_cuda,
    set_seed,
)

CONF = 0.25
IOU_MATCH = 0.5
N_FAILURES = 6


# ── Ground truth / matching helpers ─────────────────────────────────────────

def load_gt(label_file: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    boxes = []
    if label_file.exists():
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c, xc, yc, bw, bh = line.split()
                xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
                boxes.append((int(c), np.array([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2])))
    return boxes


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt, preds, conf_min: float):
    """Greedy same-class matching at IoU 0.5. Returns (tp flags per kept pred, n_fn)."""
    preds = sorted([p for p in preds if p[1] >= conf_min], key=lambda p: -p[1])
    used = set()
    tp_flags = []
    for cls, conf, box in preds:
        best, best_j = 0.0, None
        for j, (gcls, gbox) in enumerate(gt):
            if j in used or gcls != cls:
                continue
            v = iou(box, gbox)
            if v > best:
                best, best_j = v, j
        if best >= IOU_MATCH:
            used.add(best_j)
            tp_flags.append((cls, conf, True))
        else:
            tp_flags.append((cls, conf, False))
    return tp_flags, len(gt) - len(used)


def ap50(records: list[tuple[float, bool]], n_gt: int) -> float | None:
    """All-point interpolated AP from (confidence, is_tp) records."""
    if n_gt == 0:
        return None
    if not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([not r[1] for r in records])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def predict_all(model, images: list[Path], class_map=None) -> dict[str, list]:
    """{image name: [(canonical class id, conf, xyxy)]} at conf >= 0.001 (for AP)."""
    out = {}
    for i in range(0, len(images), 16):
        batch = images[i:i + 16]
        results = model.predict([str(p) for p in batch], conf=0.001, iou=0.7, imgsz=640, device=0, verbose=False)
        for p, r in zip(batch, results):
            preds = []
            for b in r.boxes:
                cls = int(b.cls.item())
                if class_map is not None:
                    if cls not in class_map:
                        continue
                    cls = CLASS_NAMES.index(class_map[cls])
                preds.append((cls, float(b.conf.item()), b.xyxy[0].cpu().numpy()))
            out[p.name] = preds
    return out


def simple_ap50(images, preds, classes: list[str]) -> dict:
    from PIL import Image

    per_class = {}
    for name in classes:
        cid = CLASS_NAMES.index(name)
        records, n_gt = [], 0
        for p in images:
            w, h = Image.open(p).size
            gt = [g for g in load_gt(p.parent.parent / "labels" / f"{p.stem}.txt", w, h) if g[0] == cid]
            n_gt += len(gt)
            flags, _ = match(gt, [x for x in preds[p.name] if x[0] == cid], 0.001)
            records += [(conf, tp) for _, conf, tp in flags]
        per_class[name] = {"AP50": None if ap50(records, n_gt) is None else round(ap50(records, n_gt), 4), "instances": n_gt}
    return per_class


def draw_failure(img_path: Path, gt, preds, out_path: Path, fn: int, fp: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(img_path).convert("RGB")
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    left, right = img.copy(), img.copy()
    dl, dr = ImageDraw.Draw(left), ImageDraw.Draw(right)
    for cls, box in gt:
        dl.rectangle(box.tolist(), outline="#16A34A", width=3)
        dl.text((box[0] + 2, max(0, box[1] - 18)), CLASS_NAMES[cls], fill="#16A34A", font=font)
    for cls, conf, box in preds:
        if conf >= CONF:
            dr.rectangle(box.tolist(), outline="#DC2626", width=3)
            dr.text((box[0] + 2, max(0, box[1] - 18)), f"{CLASS_NAMES[cls]} {conf:.2f}", fill="#DC2626", font=font)
    dl.text((6, 6), "GROUND TRUTH", fill="white", font=font)
    dr.text((6, 6), f"PREDICTION (conf>={CONF})  FN={fn} FP={fp}", fill="white", font=font)
    canvas = Image.new("RGB", (img.width * 2 + 10, img.height), "black")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (img.width + 10, 0))
    canvas.save(out_path, quality=90)


def main() -> None:
    hardware = require_cuda()
    set_seed(SEED)
    allow_trusted_checkpoints()
    from PIL import Image
    from ultralytics import YOLO

    best = VISION_MODEL_DIR / "best.pt"
    info_path = VISION_MODEL_DIR / "training_info.json"
    if not best.exists() or not info_path.exists():
        raise SystemExit("ERROR: models/vision/best.pt or training_info.json missing. Run train_yolo.py first.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    data_yaml = PROCESSED_VISION_DIR / "data.yaml"
    stats = json.loads((PROCESSED_VISION_DIR / "stats.json").read_text(encoding="utf-8"))
    figures_dir = VISION_MODEL_DIR / "figures"
    failures_dir = figures_dir / "failures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(best))

    # ── 1. Official test metrics ─────────────────────────────────────────────
    val_name = "yolov8n_flood_test"
    res = model.val(data=str(data_yaml), split="test", imgsz=640, batch=8, device=0, plots=True,
                    project=str(RUNS_DIR), name=val_name, exist_ok=True, verbose=True)
    test_instances = stats["instances"]["test"]
    per_class = {}
    for name in CLASS_NAMES:
        cid = CLASS_NAMES.index(name)
        if cid in list(res.box.ap_class_index):
            k = list(res.box.ap_class_index).index(cid)
            p, r, a50, _ = res.box.class_result(k)
            per_class[name] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                               "AP50": round(float(a50), 4), "instances": test_instances[name]}
        else:
            per_class[name] = {"precision": None, "recall": None, "AP50": None,
                               "instances": test_instances[name], "note": "no test instances; not evaluable"}
    metrics_test = {"mAP50": round(float(res.box.map50), 4), "mAP50_95": round(float(res.box.map), 4),
                    "per_class": per_class}
    print("TEST:", json.dumps(metrics_test, indent=2))

    val_dir = RUNS_DIR / val_name
    for src, dst in [("confusion_matrix.png", "confusion_matrix.png"),
                     ("confusion_matrix_normalized.png", "confusion_matrix_normalized.png"),
                     ("PR_curve.png", "PR_curve.png"), ("F1_curve.png", "F1_curve.png")]:
        if (val_dir / src).exists():
            shutil.copy2(val_dir / src, figures_dir / dst)

    # ── 2. Failure cases + like-for-like AP50 ────────────────────────────────
    test_images = sorted((PROCESSED_VISION_DIR / "test" / "images").iterdir())
    preds = predict_all(model, test_images)
    ranked = []
    for p in test_images:
        w, h = Image.open(p).size
        gt = load_gt(p.parent.parent / "labels" / f"{p.stem}.txt", w, h)
        flags, fn = match(gt, preds[p.name], CONF)
        fp = sum(not t for _, _, t in flags)
        ranked.append((fn + fp, fn, fp, p, gt))
    ranked.sort(key=lambda x: (-x[0], x[3].name))
    if failures_dir.exists():
        shutil.rmtree(failures_dir)
    failures_dir.mkdir(parents=True)
    # One failure per source image: augmented copies of the same frame would otherwise fill all six slots
    from prepare_vision import source_key
    worst, seen = [], set()
    for row in ranked:
        if source_key(row[3]) not in seen:
            seen.add(source_key(row[3]))
            worst.append(row)
        if len(worst) == N_FAILURES:
            break
    failures = []
    for i, (err, fn, fp, p, gt) in enumerate(worst, 1):
        out = failures_dir / f"failure_{i}.jpg"
        draw_failure(p, gt, preds[p.name], out, fn, fp)
        failures.append({"file": f"figures/failures/{out.name}", "image": p.name, "false_negatives": fn,
                         "false_positives": fp})
    print("Worst failures:", failures)

    evaluable = [c for c in CLASS_NAMES if test_instances[c] > 0]
    finetuned_simple = simple_ap50(test_images, preds, evaluable)

    # ── 3. COCO baseline (person and vehicle only) ──────────────────────────
    coco = YOLO(str(BASE_YOLO_WEIGHTS))
    coco_preds = predict_all(coco, test_images, class_map=COCO_TO_CANONICAL)
    coco_simple = simple_ap50(test_images, coco_preds, [c for c in ["person", "vehicle"] if c in evaluable])
    baselines = {
        "coco_yolov8n": {
            "description": "COCO-pretrained yolov8n.pt, no fine-tuning; COCO person -> person, "
                           "car/motorcycle/bus/truck -> vehicle. Cannot detect person_in_water or person_in_vehicle.",
            "method": "simple all-point AP50, greedy same-class matching at IoU 0.5 (eval_vision.py)",
            "per_class": coco_simple,
        },
        "finetuned_same_method": {
            "description": "The fine-tuned model scored with the same simple AP50 code, for a like-for-like comparison.",
            "method": "simple all-point AP50, greedy same-class matching at IoU 0.5 (eval_vision.py)",
            "per_class": finetuned_simple,
        },
    }
    print("Baselines:", json.dumps(baselines, indent=2))

    # ── 4. metrics.json ──────────────────────────────────────────────────────
    raw_yaml = yaml.safe_load((PROCESSED_VISION_DIR.parent.parent / "raw" / "flood_yolo" / "data.yaml").read_text())
    notes = [
        "Test split carved from the export's train/ by source stem (10%, seed 42); see DATA.md.",
        "Roboflow's valid/ overlaps train/, so validation metrics used for early stopping are optimistic; "
        "only test metrics are reported here.",
    ]
    missing = stats.get("canonical_classes_without_source", [])
    if missing:
        notes.insert(0, f"The dataset export has no instances of {missing}; the model cannot detect them and "
                        f"their metrics are null (not evaluable), not zero.")
    metrics = {
        "model_name": "mdris-yolov8n-flood",
        "task": "object_detection",
        "base_model": "yolov8n.pt (COCO)",
        "trained_at": info["trained_at"],
        "hardware": info["hardware"],
        "train_time_minutes": info["train_time_minutes"],
        "dataset": {
            "name": VISION_DATASET["name"] + f" v{raw_yaml.get('roboflow', {}).get('version', '?')}",
            "url": VISION_DATASET["url"],
            "licence": raw_yaml.get("roboflow", {}).get("license", "unknown"),
            "splits": stats["splits"],
        },
        "classes": CLASS_NAMES,
        "hyperparameters": {"imgsz": 640, "epochs": 60, "epochs_run": info["epochs_run"], "patience": 15,
                            "batch": info["batch"], "workers": info["workers"], "optimizer": "auto", "amp": True,
                            "seed": SEED},
        "metrics": {"test": metrics_test},
        "baselines": baselines,
        "failure_cases": failures,
        "figures": ["figures/confusion_matrix.png", "figures/PR_curve.png", "figures/F1_curve.png",
                    "figures/results.png"] + [f["file"] for f in failures],
        "evaluated_on": hardware,
        "notes": notes,
    }
    (VISION_MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {VISION_MODEL_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
