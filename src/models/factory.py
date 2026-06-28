"""
The model factory -- the ONE place that decides which detector to use.

This is the heart of "how to switch from one model to another". Every part of the
system calls `load_detector()` and gets back something that obeys the BaseDetector
interface. To switch models you set the env var IVP_MODEL_BACKEND (or edit
config.MODEL_BACKEND). Nothing else in the codebase has to change.

    IVP_MODEL_BACKEND=autoencoder  -> trained autoencoder (recommended)
    IVP_MODEL_BACKEND=classifier   -> trained ResNet18 classifier
    IVP_MODEL_BACKEND=dummy        -> numpy baseline (no PyTorch, used by CI/tests)
    IVP_MODEL_BACKEND=auto         -> use a trained model if present, else dummy
"""

from __future__ import annotations

from pathlib import Path

from src import config
from src.models.base import BaseDetector
from src.models.dummy_detector import DummyDetector


def _autoencoder_path() -> Path:
    return config.MODELS_DIR / f"autoencoder_{config.CATEGORY}.pt"


def _classifier_path() -> Path:
    return config.MODELS_DIR / f"classifier_{config.CATEGORY}.pt"


def load_detector(backend: str | None = None) -> BaseDetector:
    """
    Return a ready-to-use detector. `backend` overrides config if given.
    Falls back to the dummy model gracefully so the system always starts.
    """
    backend = (backend or config.MODEL_BACKEND).lower()

    if backend == "dummy":
        return DummyDetector(threshold=config.DEFAULT_THRESHOLD)

    if backend in ("autoencoder", "auto") and _autoencoder_path().exists():
        from src.models.autoencoder import AutoencoderDetector

        return AutoencoderDetector.load(_autoencoder_path())

    if backend in ("classifier", "auto") and _classifier_path().exists():
        from src.models.classifier import ClassifierDetector

        return ClassifierDetector.load(_classifier_path())

    # Explicit request for a trained model that isn't there yet -> clear guidance.
    if backend == "autoencoder":
        raise FileNotFoundError(
            f"No trained autoencoder at {_autoencoder_path()}.\n"
            "Train one first:  python -m src.training.train_anomaly"
        )
    if backend == "classifier":
        raise FileNotFoundError(
            f"No trained classifier at {_classifier_path()}.\n"
            "Train one first:  python -m src.training.train_classifier"
        )

    # backend == "auto" and nothing trained yet -> dummy keeps the demo alive.
    return DummyDetector(threshold=config.DEFAULT_THRESHOLD)
