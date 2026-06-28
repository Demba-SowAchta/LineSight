"""
Train the ResNet18 good/defect classifier (Transfer Learning).

This is the ALTERNATIVE path from the course Quick-Start Guide. It needs labelled
good and defect images. We build the labels from MVTec LOCO's test folders:
'good' images are class 0, anomaly images (logical+structural) are class 1.

NOTE: because MVTec LOCO's 'train' split is good-only, we carve a labelled
train/val split out of the labelled TEST images here for a simple demo. For a
rigorous result, keep a held-out test set the classifier never sees (the
evaluate.py script can be pointed at such a split).

RUN:  python -m src.training.train_classifier --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import random

import numpy as np

from src import config


def _labelled_paths(category: str) -> list[tuple[str, int]]:
    """Return (path, label) pairs from the manifest: good=0, anomaly=1."""
    manifest = config.DATA_DIR / f"manifest_{category}.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found. Run:  python -m src.training.prepare_data "
            f"--category {category}"
        )
    pairs = []
    with manifest.open() as f:
        for row in csv.DictReader(f):
            if row["split"] != "test":
                continue
            label = 0 if row["label"] == "good" else 1
            pairs.append((row["path"], label))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ResNet18 classifier.")
    parser.add_argument("--category", default=config.CATEGORY)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from src.models.classifier import ClassifierDetector, build_classifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    pairs = _labelled_paths(args.category)
    random.seed(42)
    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_fraction)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    class DS(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, label = self.items[i]
            img = Image.open(path).convert("RGB").resize((224, 224))
            arr = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
            x = torch.from_numpy(arr).permute(2, 0, 1).float()
            return x, label

    train_loader = DataLoader(DS(train_pairs), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DS(val_pairs), batch_size=args.batch_size)

    model = build_classifier(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)  # train final layer
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        # quick validation accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total if total else 0.0
        print(f"epoch {epoch:3d}/{args.epochs}  val_acc={acc:.3f}")

    detector = ClassifierDetector(model=model, device=device, class_names=["good", "defect"])
    out = config.MODELS_DIR / f"classifier_{args.category}.pt"
    detector.save(out)
    print(f"Saved model -> {out}")
    print("Switch to it with:  export IVP_MODEL_BACKEND=classifier")


if __name__ == "__main__":
    main()
