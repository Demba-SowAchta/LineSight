"""
ClassifierDetector -- the ALTERNATIVE model (the Quick-Start Guide's path).

WHEN TO USE THIS INSTEAD OF THE AUTOENCODER:
If you DO have labelled good/defect images (MVTec LOCO's test folders are
labelled by defect type), a supervised classifier is simpler to reason about and
often more accurate on the defects it was trained on. The trade-off: it only
recognises defect categories it saw during training, so a brand-new defect type
may slip through. The autoencoder is the safer default for open-ended defects;
this classifier is the better choice when defects are well-defined and labelled.

HOW IT WORKS (Transfer Learning, exactly as the course guide recommends):
We take ResNet18 already trained on millions of ImageNet images, freeze its
feature extractor, and train only a new final layer for our 2 classes
(good / defect). This needs little data and trains in minutes.

It still implements the SAME BaseDetector interface, so it is fully swappable
with the autoencoder and the dummy model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.base import BaseDetector, DetectionResult


def _torch():
    try:
        return __import__("torch")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the classifier model.\n"
            "Install it with:  pip install torch torchvision"
        ) from exc


def build_classifier(num_classes: int = 2):
    """
    Build a ResNet18 with a fresh final layer and a frozen backbone.
    `num_classes=2` -> good vs defect. Increase it for per-defect-type classes.
    """
    _torch()
    import torch.nn as nn
    from torchvision import models

    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in net.parameters():  # freeze the pretrained backbone
        param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, num_classes)  # train only this layer
    return net


class ClassifierDetector(BaseDetector):
    name = "resnet18-classifier-v1"

    # By convention class index 1 = "defect". We report its probability as the score.
    def __init__(
        self,
        model=None,
        threshold: float = 0.5,
        image_size: int = 224,
        device: str = "cpu",
        class_names: list[str] | None = None,
    ):
        self.torch = _torch()
        self.model = model
        self.threshold = threshold
        self.image_size = image_size
        self.device = device
        self.class_names = class_names or ["good", "defect"]
        if self.model is not None:
            self.model.eval().to(device)

    def _to_tensor(self, image: np.ndarray):
        """Preprocess exactly as ImageNet expects (resize + standard normalisation)."""
        from PIL import Image

        torch = self.torch
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        pil = pil.resize((self.image_size, self.image_size))
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def predict(self, image: np.ndarray) -> DetectionResult:
        torch = self.torch
        x = self._to_tensor(image)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        defect_prob = float(probs[1])  # probability of the 'defect' class
        return DetectionResult(
            score=defect_prob,
            is_anomaly=defect_prob >= self.threshold,
            threshold=self.threshold,
            heatmap=None,  # plain classifier has no localisation map
            extra={"probs": {n: float(p) for n, p in zip(self.class_names, probs)}},
        )

    def save(self, path: Path) -> None:
        torch = self.torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        meta = {
            "threshold": self.threshold,
            "image_size": self.image_size,
            "class_names": self.class_names,
            "name": self.name,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "ClassifierDetector":
        torch = _torch()
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        model = build_classifier(num_classes=len(meta["class_names"]))
        model.load_state_dict(torch.load(path, map_location=device))
        return cls(
            model=model,
            threshold=meta["threshold"],
            image_size=meta["image_size"],
            device=device,
            class_names=meta["class_names"],
        )
