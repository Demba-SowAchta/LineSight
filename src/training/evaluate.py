"""
Evaluate a trained detector on the MVTec LOCO test set.

PRODUCES exactly what the course rubric asks for:
  - Confusion matrix (TP / FP / TN / FN)
  - Precision, Recall, F1
  - Accuracy
  - AUROC (how well the score separates good from defect, independent of threshold)
  - A confusion-matrix PNG and an ROC-curve PNG saved to artifacts/eval/

The detector is loaded through the SAME factory the live system uses, so you are
evaluating exactly what you deploy.

RUN:  python -m src.training.evaluate                       # uses IVP_MODEL_BACKEND
      IVP_MODEL_BACKEND=autoencoder python -m src.training.evaluate
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src import config
from src.models import load_detector
from src.utils.images import load_image


def _test_items(category: str) -> list[tuple[str, int]]:
    """(path, label) for the test split. label: 0=good, 1=defect."""
    manifest = config.DATA_DIR / f"manifest_{category}.csv"
    items = []
    with manifest.open() as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                items.append((row["path"], 0 if row["label"] == "good" else 1))
    return items


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Area under the ROC curve via the rank-sum (Mann-Whitney U) identity.
    No scikit-learn needed. 1.0 = perfect separation, 0.5 = random.
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained detector.")
    parser.add_argument("--category", default=config.CATEGORY)
    args = parser.parse_args()

    detector = load_detector()
    print(f"Evaluating model: {detector.name} (threshold={detector.threshold:.3f})")

    items = _test_items(args.category)
    if not items:
        raise SystemExit("No test items. Run prepare_data and check the dataset path.")

    scores, labels, preds = [], [], []
    for path, label in items:
        res = detector.predict(load_image(path))
        scores.append(res.score)
        labels.append(label)
        preds.append(1 if res.is_anomaly else 0)

    scores = np.array(scores)
    labels = np.array(labels)
    preds = np.array(preds)

    # Confusion matrix (positive class = defect)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(items)
    auroc = _auroc(scores, labels)

    print("\n================ RESULTS ================")
    print(f"Images tested : {len(items)}  (good={int((labels==0).sum())}, defect={int((labels==1).sum())})")
    print(f"Confusion     : TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Precision     : {precision:.3f}")
    print(f"Recall        : {recall:.3f}   (share of defects we catch)")
    print(f"F1 score      : {f1:.3f}")
    print(f"Accuracy      : {accuracy:.3f}")
    print(f"AUROC         : {auroc:.3f}")
    print("=========================================\n")

    # Save plots (best-effort; skip silently if matplotlib is absent).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = config.PROJECT_ROOT / "artifacts" / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["pred good", "pred defect"])
        ax.set_yticks([0, 1], ["true good", "true defect"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_title(f"Confusion matrix ({detector.name})")
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_{args.category}.png", dpi=120)

        # ROC curve
        thresholds = np.linspace(0, 1, 100)
        tpr = [((scores >= t)[labels == 1]).mean() if (labels == 1).any() else 0 for t in thresholds]
        fpr = [((scores >= t)[labels == 0]).mean() if (labels == 0).any() else 0 for t in thresholds]
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
        ax2.plot([0, 1], [0, 1], "--", color="gray")
        ax2.set_xlabel("False positive rate")
        ax2.set_ylabel("True positive rate")
        ax2.set_title("ROC curve")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(out_dir / f"roc_{args.category}.png", dpi=120)
        print(f"Plots saved -> {out_dir}")
    except ImportError:
        print("(matplotlib not installed -> skipped plots; metrics above are complete)")


if __name__ == "__main__":
    main()
