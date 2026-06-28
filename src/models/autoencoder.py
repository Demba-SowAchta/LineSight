"""
AutoencoderDetector -- the RECOMMENDED model for assembly-error detection.

WHY AN AUTOENCODER :
In a factory you can photograph thousands of GOOD parts, but defects are rare and
endlessly varied (a new scratch, a new missing screw, a colour you never saw).
You cannot label them all. So instead of learning "what defects look like", we
learn "what GOOD looks like" and flag anything that does not fit.

A convolutional autoencoder is trained ONLY on good images to compress an image
into a small code and rebuild it. Trained on good parts, it rebuilds good parts
well (low error) but rebuilds unfamiliar defects poorly (high error). The
per-pixel rebuild error is both the anomaly SCORE and a HEATMAP showing where the
defect is. This is unsupervised: no defect labels needed for training.

TRADE-OFFS (be honest in the report):
+ No defect labels needed; detects defect types never seen in training.
+ Gives a localisation heatmap for free -> great for the live demo.
- Less precise than a fully-supervised detector when you DO have lots of labels.
- Sensitive to lighting changes -> fix the lighting (see docs/03_ai_pipeline.md).
For stronger results, anomalib's PatchCore/EfficientAD are drop-in upgrades that
implement this same BaseDetector interface (see docs/03_ai_pipeline.md).

This file imports torch lazily so the rest of the project still runs without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.base import BaseDetector, DetectionResult


def _torch():
    """Import torch only when actually needed, with a friendly error if missing."""
    try:
        import torch  # noqa: F401

        return __import__("torch")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the autoencoder model.\n"
            "Install it with:  pip install torch torchvision"
        ) from exc


def build_autoencoder(image_size: int = 256):
    """
    Build the autoencoder network.

    Encoder: shrinks 3x256x256 down to a small feature map (the 'code').
    Decoder: mirrors the encoder to rebuild the original image.
    Kept deliberately small so it trains in minutes on a free Colab T4 GPU.
    """
    _torch()
    import torch.nn as nn

    class ConvAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Encoder: each block halves the spatial size and adds channels.
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1),
                nn.ReLU(True),  # 128
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(True),  # 64
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU(True),  # 32
                nn.Conv2d(128, 128, 4, stride=2, padding=1),
                nn.ReLU(True),  # 16
            )
            # Decoder: each block doubles the spatial size back up.
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
                nn.ReLU(True),  # 32
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(True),  # 64
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(True),  # 128
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
                nn.Sigmoid(),  # 256
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return ConvAutoencoder()


class AutoencoderDetector(BaseDetector):
    """Wraps a trained autoencoder so it satisfies the BaseDetector interface."""

    name = "autoencoder-v1"

    def __init__(
        self,
        model=None,
        threshold: float = 0.5,
        image_size: int = 256,
        device: str = "cpu",
        score_min: float = 0.0,
        score_max: float = 1.0,
    ):
        self.torch = _torch()
        self.model = model
        self.threshold = threshold
        self.image_size = image_size
        self.device = device
        # score_min/max normalise raw reconstruction error to a friendly 0..1 range.
        self.score_min = score_min
        self.score_max = score_max
        if self.model is not None:
            self.model.eval().to(device)

    # -- preprocessing --------------------------------------------------------
    def _to_tensor(self, image: np.ndarray):
        """HxWx3 uint8 RGB -> normalised 1x3xHxW float tensor."""
        from PIL import Image

        torch = self.torch
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        pil = pil.resize((self.image_size, self.image_size))
        arr = np.asarray(pil, dtype=np.float32) / 255.0  # 0..1
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
        return tensor.to(self.device)

    # -- inference ------------------------------------------------------------
    def predict(self, image: np.ndarray) -> DetectionResult:
        torch = self.torch
        x = self._to_tensor(image)
        with torch.no_grad():
            recon = self.model(x)
            # Per-pixel squared error, averaged over colour channels -> error map.
            err_map = ((recon - x) ** 2).mean(dim=1).squeeze(0).cpu().numpy()

        raw_score = float(err_map.mean())  # whole-image anomaly score
        score = self._normalise(raw_score)  # squash to 0..1
        heatmap = self._normalise_map(err_map)  # 0..1 localisation map

        return DetectionResult(
            score=score,
            is_anomaly=score >= self.threshold,
            threshold=self.threshold,
            heatmap=heatmap,
            extra={"raw_score": raw_score},
        )

    def _normalise(self, raw: float) -> float:
        span = max(self.score_max - self.score_min, 1e-9)
        return float(np.clip((raw - self.score_min) / span, 0.0, 1.0))

    @staticmethod
    def _normalise_map(m: np.ndarray) -> np.ndarray:
        lo, hi = float(m.min()), float(m.max())
        return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)

    # -- persistence ----------------------------------------------------------
    def save(self, path: Path) -> None:
        """Save weights AND calibration (threshold + score range) side by side."""
        torch = self.torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        meta = {
            "threshold": self.threshold,
            "image_size": self.image_size,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "name": self.name,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "AutoencoderDetector":
        """Rebuild a ready-to-use detector from saved weights + calibration."""
        torch = _torch()
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        model = build_autoencoder(meta["image_size"])
        model.load_state_dict(torch.load(path, map_location=device))
        return cls(
            model=model,
            threshold=meta["threshold"],
            image_size=meta["image_size"],
            device=device,
            score_min=meta["score_min"],
            score_max=meta["score_max"],
        )
