"""
Small image helpers shared across the project. Pure Pillow + numpy, no heavy deps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load any image file from disk as an HxWx3 uint8 RGB numpy array."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Save an HxWx3 uint8 array to disk, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
    return path


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """
    Blend a 0..1 anomaly heatmap (red = hot) over the original image.
    The heatmap is resized to match the image, so coarse maps still overlay cleanly.
    This is what produces the striking 'where is the defect' picture in the demo.
    """
    h, w = image.shape[:2]
    hm = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8)).resize((w, h))
    hm = np.asarray(hm, dtype=np.float32) / 255.0

    # Map intensity to a red overlay (hot spots become red).
    color = np.zeros((h, w, 3), dtype=np.float32)
    color[..., 0] = hm  # red channel follows the heat

    base = np.asarray(image, dtype=np.float32) / 255.0
    blended = (1 - alpha * hm[..., None]) * base + (alpha * hm[..., None]) * color
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
