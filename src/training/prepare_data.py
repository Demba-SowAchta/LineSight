"""
Data preparation for MVTec LOCO AD.

WHAT MVTec LOCO LOOKS LIKE ON DISK (after you unzip it):
    data/mvtec_loco/<category>/
        train/good/          <- only GOOD images (this is by design)
        validation/good/     <- good images for calibrating the threshold
        test/good/           <- good images for testing
        test/logical_anomalies/     <- defects: a rule is broken
        test/structural_anomalies/  <- defects: a physical problem
        ground_truth/...            <- pixel masks (not needed for our models)

WHY ONLY GOOD IN TRAIN: the dataset is built for ANOMALY DETECTION. You train on
"good" and detect deviations. That matches reality -- you have many good parts and
few, varied defects. Our autoencoder uses exactly this split.

WHAT THIS SCRIPT DOES: it scans the category folder and writes a single CSV
"manifest" listing every image with its split (train/val/test) and label
(good/logical/structural). Downstream scripts read the manifest instead of walking
folders, which keeps training and evaluation simple and reproducible.

RUN:  python -m src.training.prepare_data            # uses config.CATEGORY
      python -m src.training.prepare_data --category juice_bottle
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src import config

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_manifest(category: str) -> list[dict]:
    """Walk the category folder and return one row per image."""
    root = config.DATA_DIR / category
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}\n"
            "Download MVTec LOCO AD and unzip it there "
            "(see scripts/download_data.md)."
        )

    rows: list[dict] = []

    def add(split: str, label: str, folder: Path) -> None:
        if not folder.exists():
            return
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({"path": str(p), "split": split, "label": label})

    # MVTec LOCO uses 'validation'; some mirrors use 'val'. Handle both.
    val_dir = root / "validation"
    if not val_dir.exists():
        val_dir = root / "val"

    add("train", "good", root / "train" / "good")
    add("val", "good", val_dir / "good")
    add("test", "good", root / "test" / "good")
    add("test", "logical", root / "test" / "logical_anomalies")
    add("test", "structural", root / "test" / "structural_anomalies")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MVTec LOCO manifest CSV.")
    parser.add_argument("--category", default=config.CATEGORY)
    args = parser.parse_args()

    rows = build_manifest(args.category)
    out = config.DATA_DIR / f"manifest_{args.category}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "split", "label"])
        writer.writeheader()
        writer.writerows(rows)

    # Print a quick summary so you can sanity-check the split counts.
    from collections import Counter
    by_split = Counter(r["split"] for r in rows)
    by_label = Counter(f"{r['split']}/{r['label']}" for r in rows)
    print(f"Manifest written: {out}  ({len(rows)} images)")
    print("By split :", dict(by_split))
    print("By bucket:", dict(by_label))


if __name__ == "__main__":
    main()
