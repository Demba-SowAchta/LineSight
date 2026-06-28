"""
The detector interface.

Every model in this project -- the numpy baseline, the autoencoder, the
classifier -- implements the SAME three methods below. This is what makes models
"swappable": the rest of the system (agents, API, app) only ever talks to this
interface, never to a specific model. To switch models you change ONE line of
config (IVP_MODEL_BACKEND); no other code changes.

The contract:

    detector.name        -> str, e.g. "autoencoder-v1"  (written into the DB)
    detector.predict(img)-> DetectionResult

A DetectionResult always carries:
    score    : float   higher = more anomalous (0..1 after normalisation)
    is_anomaly: bool    score >= threshold
    heatmap  : optional 2D array highlighting where the anomaly is (for the demo)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class DetectionResult:
    """The single, uniform output of every detector."""

    score: float  # anomaly score, higher = worse
    is_anomaly: bool  # final yes/no after thresholding
    threshold: float  # threshold used (for the DB audit trail)
    heatmap: Optional[np.ndarray] = None  # 2D map of where the defect is
    extra: Optional[dict[str, Any]] = None  # any model-specific details

    @property
    def confidence(self) -> float:
        """
        How far the score is from the decision boundary, in 0..1.
        Near the threshold = unsure; far from it = confident.
        """
        return float(
            min(1.0, abs(self.score - self.threshold) / max(self.threshold, 1e-6))
        )


class BaseDetector:
    """
    Base class. A real detector subclasses this and implements `predict`.

    Subclasses set `self.threshold` (the score above which a part is rejected)
    and `self.name` (written into every database record for traceability).
    """

    name: str = "base"
    threshold: float = 0.5

    def predict(self, image: np.ndarray) -> DetectionResult:
        """
        Run inference on a single image.

        `image` is an HxWx3 uint8 RGB numpy array. Return a DetectionResult.
        Subclasses MUST override this.
        """
        raise NotImplementedError("Subclasses must implement predict()")

    # Convenience: let callers do `detector(image)` instead of `detector.predict(image)`
    def __call__(self, image: np.ndarray) -> DetectionResult:
        return self.predict(image)
