"""
Fine-tune YOLOv8n on the flood dataset (docs/MDRIS_CONTEXT.md section 7.2)
============================================================================
Base weights backend/yolov8n.pt (COCO) -> datasets/processed/flood_yolo (4 classes).
Settings are fixed by the spec; the only allowed changes are the out-of-memory
fallback (--batch 4) and --workers 0 if the Windows dataloader hangs.

Afterwards best.pt and the standard figures are copied to models/vision/, and
models/vision/training_info.json records time, epochs and hardware for
eval_vision.py, which writes metrics.json.

Usage (run in the background, logging to a file):
    python -u training/train_yolo.py > training/runs/yolo_train.log 2>&1
"""

import argparse
import csv
import json
import shutil
import time

from common import (
    BASE_YOLO_WEIGHTS,
    PROCESSED_VISION_DIR,
    RUNS_DIR,
    SEED,
    VISION_MODEL_DIR,
    allow_trusted_checkpoints,
    now_iso,
    require_cuda,
    set_seed,
)

RUN_NAME = "yolov8n_flood"
FIGURES = ["results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, choices=[8, 4], help="8, or 4 after an out-of-memory error")
    parser.add_argument("--workers", type=int, default=2, choices=[2, 0], help="2, or 0 if the dataloader hangs")
    parser.add_argument("--freeze", type=int, default=0, choices=[0, 10],
                        help="0 (spec default) or 10 = freeze the backbone (experiment E1, section 7.2)")
    parser.add_argument("--promote", action="store_true",
                        help="copy best.pt to models/vision/ (always on for freeze 0; opt-in for experiments)")
    args = parser.parse_args()
    run_name = RUN_NAME if args.freeze == 0 else f"{RUN_NAME}_freeze{args.freeze}"
    promote = args.freeze == 0 or args.promote

    hardware = require_cuda()
    set_seed(SEED)
    allow_trusted_checkpoints()
    from ultralytics import YOLO

    data_yaml = PROCESSED_VISION_DIR / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"ERROR: {data_yaml} not found. Run training/prepare_vision.py first.")

    print(f"[{now_iso()}] Training YOLOv8n on {hardware} | batch={args.batch} workers={args.workers} "
          f"freeze={args.freeze} run={run_name} promote={promote}", flush=True)
    start = time.time()
    model = YOLO(str(BASE_YOLO_WEIGHTS))
    model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=60,
        patience=15,
        batch=args.batch,
        workers=args.workers,
        amp=True,
        optimizer="auto",
        seed=SEED,
        deterministic=True,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        device=0,
        freeze=args.freeze or None,
    )
    minutes = (time.time() - start) / 60

    run_dir = RUNS_DIR / run_name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"ERROR: training finished but {best} does not exist.")
    if not promote:
        print(f"[{now_iso()}] Done in {minutes:.1f} min. NOT promoted; checkpoint stays at {best}", flush=True)
        return

    figures_dir = VISION_MODEL_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, VISION_MODEL_DIR / "best.pt")
    for fig in FIGURES:
        if (run_dir / fig).exists():
            shutil.copy2(run_dir / fig, figures_dir / (fig if fig == "results.png" else f"val_{fig}"))

    with (run_dir / "results.csv").open(encoding="utf-8") as f:
        epochs_run = sum(1 for _ in csv.DictReader(f))

    info = {
        "trained_at": now_iso(),
        "hardware": f"NVIDIA {hardware}" if not hardware.startswith("NVIDIA") else hardware,
        "train_time_minutes": round(minutes, 1),
        "epochs_run": epochs_run,
        "batch": args.batch,
        "workers": args.workers,
        "freeze": args.freeze,
        "run_dir": run_dir.relative_to(RUNS_DIR.parent.parent).as_posix(),
    }
    (VISION_MODEL_DIR / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[{now_iso()}] Done in {minutes:.1f} min, {epochs_run} epochs. best.pt -> {VISION_MODEL_DIR / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
