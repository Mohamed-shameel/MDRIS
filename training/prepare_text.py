"""
Prepare the flood urgency text dataset (docs/MDRIS_CONTEXT.md section 5.2)
============================================================================
datasets/raw/humaid/  ->  datasets/processed/flood_text/{train,val,test}.csv

Locked order (section 5.2.1):
1. Load every split of every event folder whose name contains "flood";
   keep tweet_id, tweet_text, class_label, plus event (folder name).
2. Drop dont_know_cant_judge and empty texts.
3. Normalize: strip a leading "RT @x:", URLs -> URL, @mentions -> @USER, collapse whitespace.
4. Deduplicate on normalized text BEFORE splitting (retweets would leak).
5. Map HumAID category -> urgency (section 5.2.3). Unknown category -> stop.
6. Stratified 70/15/15 split on urgency, seed 42.
7. Save CSVs with columns text, urgency, humaid_label, event.
8. Fewer than 1,500 rows -> stop.

Raw data is only read, never modified.

Usage:  python training/prepare_text.py
"""

import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import HUMAID_TO_URGENCY, PROCESSED_TEXT_DIR, RAW_TEXT_DIR, SEED, SPOT_CHECK_DIR, URGENCY_LABELS

REQUIRED_COLUMNS = ["tweet_id", "tweet_text", "class_label"]
DROP_LABELS = {"dont_know_cant_judge"}
MIN_ROWS = 1500
SPOT_CHECK_ROWS = 50

RT_PREFIX = re.compile(r"^\s*RT\s+@\w+:\s*", re.IGNORECASE)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = RT_PREFIX.sub("", text)
    text = URL_RE.sub("URL", text)
    text = MENTION_RE.sub("@USER", text)
    return SPACE_RE.sub(" ", text).strip()


def find_flood_files(root: Path) -> list[Path]:
    """Every .tsv whose immediate parent folder name contains 'flood' (macOS '._' files skipped)."""
    return sorted(
        p for p in root.rglob("*.tsv")
        if "flood" in p.parent.name.lower() and not p.name.startswith("._")
    )


def load(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        df = pd.read_csv(f, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            sys.exit(f"STOP: {f} is missing columns {missing}; found {list(df.columns)}")
        df = df[REQUIRED_COLUMNS].copy()
        df["event"] = f.parent.name
        df["source_file"] = f.name
        frames.append(df)
        print(f"  {f.relative_to(root_dir()).as_posix()}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def root_dir() -> Path:
    return RAW_TEXT_DIR


def distribution(df: pd.DataFrame, col: str) -> dict:
    return {k: int(v) for k, v in df[col].value_counts().sort_index().items()}


def main() -> None:
    if not RAW_TEXT_DIR.is_dir():
        sys.exit(f"STOP: {RAW_TEXT_DIR} not found. Download HumAID first (section 5.2).")

    # ── Inspect ────────────────────────────────────────────────────────────
    all_tsv = sorted(p for p in RAW_TEXT_DIR.rglob("*.tsv") if not p.name.startswith("._"))
    events = sorted({p.parent.name for p in all_tsv})
    print(f"Found {len(all_tsv)} .tsv files across {len(events)} event folders: {events}")
    files = find_flood_files(RAW_TEXT_DIR)
    if not files:
        sys.exit("STOP: no flood event .tsv files found under datasets/raw/humaid/.")
    print(f"Flood files ({len(files)}):")

    # ── 1. Load ────────────────────────────────────────────────────────────
    df = load(files)
    stats = {"raw_rows": len(df), "raw_rows_per_event": distribution(df, "event"), "files": [f.name for f in files]}

    # ── 2. Drop ────────────────────────────────────────────────────────────
    df["class_label"] = df["class_label"].str.strip()
    dropped_label = int(df["class_label"].isin(DROP_LABELS).sum())
    df = df[~df["class_label"].isin(DROP_LABELS)]
    dropped_empty = int((df["tweet_text"].str.strip() == "").sum())
    df = df[df["tweet_text"].str.strip() != ""]

    unknown = sorted(set(df["class_label"]) - set(HUMAID_TO_URGENCY))
    if unknown:
        sys.exit(f"STOP: HumAID categories not in the section 5.2.3 mapping: {unknown}")

    # ── 3. Normalize ───────────────────────────────────────────────────────
    df["text"] = df["tweet_text"].map(normalize)
    dropped_empty_after = int((df["text"] == "").sum())
    df = df[df["text"] != ""]

    # ── 4. Deduplicate on normalized text (keep first, in sorted file order) ─
    before = len(df)
    conflicting = int(df.groupby("text")["class_label"].nunique().gt(1).sum())
    df = df.drop_duplicates(subset="text", keep="first")
    removed_dupes = before - len(df)

    # ── 5. Map to urgency ──────────────────────────────────────────────────
    df = df.rename(columns={"class_label": "humaid_label"})
    df["urgency"] = df["humaid_label"].map(HUMAID_TO_URGENCY)

    print(f"\nDropped: {dropped_label} dont_know_cant_judge, {dropped_empty + dropped_empty_after} empty texts")
    print(f"Deduplicated on normalized text: removed {removed_dupes} rows "
          f"({conflicting} duplicate groups had conflicting labels; first occurrence kept)")
    print(f"Rows after cleaning: {len(df)}")

    if len(df) < MIN_ROWS:
        sys.exit(f"STOP: only {len(df)} rows remain (< {MIN_ROWS}).")

    # ── 6. Stratified 70 / 15 / 15 ─────────────────────────────────────────
    df = df[["text", "urgency", "humaid_label", "event"]].reset_index(drop=True)
    train, rest = train_test_split(df, test_size=0.30, stratify=df["urgency"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["urgency"], random_state=SEED)

    # ── 7. Save ────────────────────────────────────────────────────────────
    PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        part.to_csv(PROCESSED_TEXT_DIR / f"{name}.csv", index=False, encoding="utf-8")

    overlap = {
        "train-val": len(set(train.text) & set(val.text)),
        "train-test": len(set(train.text) & set(test.text)),
        "val-test": len(set(val.text) & set(test.text)),
    }
    stats.update({
        "dropped_dont_know": dropped_label,
        "dropped_empty": dropped_empty + dropped_empty_after,
        "duplicates_removed": removed_dupes,
        "duplicate_groups_with_conflicting_labels": conflicting,
        "rows_after_cleaning": len(df),
        "rows_per_event_after_cleaning": distribution(df, "event"),
        "urgency_distribution": distribution(df, "urgency"),
        "humaid_distribution": distribution(df, "humaid_label"),
        "splits": {k: len(v) for k, v in splits.items()},
        "urgency_per_split": {k: {u: int((v.urgency == u).sum()) for u in URGENCY_LABELS} for k, v in splits.items()},
        "event_per_split": {k: distribution(v, "event") for k, v in splits.items()},
        "text_overlap_between_splits": overlap,
        "seed": SEED,
    })
    (PROCESSED_TEXT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\nSplit sizes:", stats["splits"])
    for k, v in stats["urgency_per_split"].items():
        print(f"  {k} urgency: {v}")
    print("Urgency overall:", stats["urgency_distribution"])
    print("HumAID categories:", stats["humaid_distribution"])
    print("Events:", stats["rows_per_event_after_cleaning"])
    print("Text overlap between splits:", overlap)

    # ── Spot check ─────────────────────────────────────────────────────────
    SPOT_CHECK_DIR.mkdir(parents=True, exist_ok=True)
    sample = df.sample(n=min(SPOT_CHECK_ROWS, len(df)), random_state=SEED)[["text", "humaid_label", "urgency"]]
    sample.to_csv(SPOT_CHECK_DIR / "text.csv", index=False, encoding="utf-8")
    print(f"Spot-check: {len(sample)} rows -> {SPOT_CHECK_DIR / 'text.csv'}")


if __name__ == "__main__":
    main()
