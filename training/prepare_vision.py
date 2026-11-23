"""
Prepare the flood detection dataset (docs/MDRIS_CONTEXT.md section 5.1)
=========================================================================
datasets/raw/flood_yolo/ (Roboflow YOLOv8 export)  ->  datasets/processed/flood_yolo/

1. Read raw data.yaml, normalize class names and remap them to the four
   canonical classes. Any unknown raw class -> print it and stop.
2. Rewrite every label file with canonical ids; validate ids and coordinates.
3. If the export has no test/ split, carve 10% of train/ into test/ (seed 42),
   grouped by SOURCE image: Roboflow stores 3 augmented copies of each training
   image ("<stem>_jpg.rf.<hash>.jpg"), and splitting copies apart would leak.
4. Write data.yaml (absolute path, 4 canonical names), stats.json and a
   spot-check folder of 30 training images with boxes drawn.

Raw data is only read, never modified.

Usage:  python training/prepare_vision.py
"""

import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from common import (
    CLASS_ID,
    CLASS_NAMES,
    PROCESSED_VISION_DIR,
    RAW_TO_CANONICAL,
    RAW_VISION_DIR,
    SEED,
    SPOT_CHECK_DIR,
    normalize_class_name,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RAW_SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}
TEST_FRACTION = 0.10
SPOT_CHECK_COUNT = 30

# "<stem>_jpg.rf.<hash>.jpg" -> "<stem>"
ROBOFLOW_STEM = re.compile(r"^(?P<stem>.+?)(?:_(?:jpe?g|png|bmp|webp))?\.rf\.[0-9a-f]+$", re.IGNORECASE)


def source_key(image_path: Path) -> str:
    m = ROBOFLOW_STEM.match(image_path.stem)
    return m.group("stem") if m else image_path.stem


def list_images(split_dir: Path) -> list[Path]:
    img_dir = split_dir / "images"
    if not img_dir.is_dir():
        return []
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def label_path_for(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_raw_names(raw_yaml: Path) -> list[str]:
    data = yaml.safe_load(raw_yaml.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def convert_label(raw_label: Path, raw_to_canon_id: dict[int, int]) -> tuple[list[str], list[str]]:
    """Return (canonical label lines, problems). A missing file means 'no objects'."""
    if not raw_label.exists():
        return [], []
    lines, problems = [], []
    for n, line in enumerate(raw_label.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            problems.append(f"{raw_label.name}:{n}: expected 5 fields (class x y w h), got {len(parts)}")
            continue
        raw_id = int(float(parts[0]))
        if raw_id not in raw_to_canon_id:
            problems.append(f"{raw_label.name}:{n}: class id {raw_id} not in data.yaml")
            continue
        coords = [float(v) for v in parts[1:]]
        if any(not 0.0 <= c <= 1.0 for c in coords):
            problems.append(f"{raw_label.name}:{n}: coordinate outside [0, 1]: {coords}")
            continue
        lines.append(f"{raw_to_canon_id[raw_id]} " + " ".join(f"{c:.6f}" for c in coords))
    return lines, problems


def draw_spot_check(train_images: list[Path], out_dir: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    colors = {0: "#DC2626", 1: "#EA580C", 2: "#CA8A04", 3: "#2563EB"}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rng = random.Random(SEED)
    sample = rng.sample(train_images, min(SPOT_CHECK_COUNT, len(train_images)))
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for i, img_path in enumerate(sample, 1):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)
        n_boxes = 0
        for line in label_path_for(img_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, xc, yc, bw, bh = line.split()
            cls = int(cls)
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            box = [xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2]
            draw.rectangle(box, outline=colors[cls], width=3)
            draw.text((box[0] + 2, max(0, box[1] - 18)), CLASS_NAMES[cls], fill=colors[cls], font=font)
            n_boxes += 1
        draw.text((6, 6), f"{img_path.name}  boxes={n_boxes}", fill="white", font=font)
        img.save(out_dir / f"{i:02d}_{img_path.stem[:40]}.jpg", quality=90)
    print(f"Spot-check: {len(sample)} images with boxes -> {out_dir}")


def main() -> None:
    raw_yaml = RAW_VISION_DIR / "data.yaml"
    if not raw_yaml.exists():
        sys.exit(f"STOP: {raw_yaml} not found. Download the Roboflow YOLOv8 export first (section 5.1).")

    # ── 1. Class remap ─────────────────────────────────────────────────────
    raw_names = read_raw_names(raw_yaml)
    normalized = [normalize_class_name(n) for n in raw_names]
    unknown = [raw for raw, norm in zip(raw_names, normalized) if norm not in RAW_TO_CANONICAL]
    print(f"Raw classes ({len(raw_names)}): {raw_names}")
    if unknown:
        sys.exit(f"STOP: raw class names not in the section 5.1.1 remap table: {unknown}")
    raw_to_canon_id = {i: CLASS_ID[RAW_TO_CANONICAL[norm]] for i, norm in enumerate(normalized)}
    remap_used = {raw: RAW_TO_CANONICAL[norm] for raw, norm in zip(raw_names, normalized)}
    print(f"Remap: {remap_used}")
    missing_canonical = [c for c in CLASS_NAMES if c not in remap_used.values()]
    if missing_canonical:
        print(f"WARNING: this export provides no raw class for canonical classes {missing_canonical}; "
              f"they will have zero instances.")

    # ── 2. Collect raw splits ──────────────────────────────────────────────
    splits: dict[str, list[Path]] = {s: list_images(RAW_VISION_DIR / d) for s, d in RAW_SPLIT_DIRS.items()}
    for s, imgs in splits.items():
        print(f"Raw {s}: {len(imgs)} images")

    test_carved = False
    if not splits["test"]:
        groups: dict[str, list[Path]] = defaultdict(list)
        for p in splits["train"]:
            groups[source_key(p)].append(p)
        keys = sorted(groups)
        rng = random.Random(SEED)
        rng.shuffle(keys)
        n_test = round(len(keys) * TEST_FRACTION)
        test_keys = set(keys[:n_test])
        splits["test"] = sorted(p for k in test_keys for p in groups[k])
        splits["train"] = sorted(p for k in keys[n_test:] for p in groups[k])
        test_carved = True
        print(f"No raw test/ split: carved {n_test}/{len(keys)} source images "
              f"({len(splits['test'])} files incl. augmented copies) from train into test (seed {SEED}).")

    # Source-image overlap between splits (leakage check)
    split_keys = {s: {source_key(p) for p in imgs} for s, imgs in splits.items()}
    overlap = {
        "train-val": len(split_keys["train"] & split_keys["val"]),
        "train-test": len(split_keys["train"] & split_keys["test"]),
        "val-test": len(split_keys["val"] & split_keys["test"]),
    }
    print(f"Source-image overlap between splits: {overlap}")

    # ── 3. Write processed dataset ─────────────────────────────────────────
    if PROCESSED_VISION_DIR.exists():
        shutil.rmtree(PROCESSED_VISION_DIR)
    stats = {"splits": {}, "instances": {}, "images_without_label_file": {}, "empty_label_images": {}}
    all_problems: list[str] = []
    for split, imgs in splits.items():
        img_out = PROCESSED_VISION_DIR / split / "images"
        lbl_out = PROCESSED_VISION_DIR / split / "labels"
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        counts = Counter()
        no_label = empty = 0
        for img in imgs:
            raw_label = label_path_for(img)
            if not raw_label.exists():
                no_label += 1
            lines, problems = convert_label(raw_label, raw_to_canon_id)
            all_problems.extend(problems)
            if not lines:
                empty += 1
            for line in lines:
                counts[CLASS_NAMES[int(line.split()[0])]] += 1
            shutil.copy2(img, img_out / img.name)
            (lbl_out / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["splits"][split] = len(imgs)
        stats["instances"][split] = {c: counts.get(c, 0) for c in CLASS_NAMES}
        stats["images_without_label_file"][split] = no_label
        stats["empty_label_images"][split] = empty

    if all_problems:
        print(f"STOP: {len(all_problems)} label problems, first 20:")
        for p in all_problems[:20]:
            print("  ", p)
        sys.exit(1)

    data_yaml = {
        "path": PROCESSED_VISION_DIR.resolve().as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    (PROCESSED_VISION_DIR / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    rare = [c for c in CLASS_NAMES if stats["instances"]["train"][c] < 100]
    stats.update({
        "raw_classes": raw_names,
        "remap": remap_used,
        "canonical_classes_without_source": missing_canonical,
        "test_carved_from_train": test_carved,
        "test_fraction": TEST_FRACTION if test_carved else None,
        "source_image_overlap": overlap,
        "classes_under_100_train_instances": rare,
        "seed": SEED,
    })
    (PROCESSED_VISION_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\nImages per split:", stats["splits"])
    print("Instances per class per split:")
    for split in splits:
        print(f"  {split}: {stats['instances'][split]}")
    print("Images without a raw label file:", stats["images_without_label_file"])
    print("Images with no objects:", stats["empty_label_images"])
    print("Classes with < 100 training instances:", rare)

    draw_spot_check(sorted((PROCESSED_VISION_DIR / "train" / "images").iterdir()), SPOT_CHECK_DIR / "vision")
    print(f"\nWrote {PROCESSED_VISION_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
