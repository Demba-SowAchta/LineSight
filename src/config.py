"""
Central configuration for the Industrial Vision Platform (IVP).

Everything tunable lives here so that beginners can change one value in one place
instead of hunting through the code. Every setting can also be overridden with an
environment variable, which is how the same code runs on a laptop, in Docker, and
on a factory edge device without editing source.

Read order for any setting:  environment variable  ->  default below.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    """Return an environment variable, or the default if it is not set."""
    return os.environ.get(name, default)


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# PROJECT_ROOT is the folder that contains this repository.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Where the MVTec LOCO dataset is unpacked (see scripts/download_data.md).
DATA_DIR = Path(_env("IVP_DATA_DIR", str(PROJECT_ROOT / "data" / "mvtec_loco")))

# Where trained model weights and thresholds are stored.
MODELS_DIR = Path(_env("IVP_MODELS_DIR", str(PROJECT_ROOT / "artifacts" / "models")))

# Where inspected images are archived for traceability.
IMAGE_STORE_DIR = Path(_env("IVP_IMAGE_STORE", str(PROJECT_ROOT / "artifacts" / "images")))

# SQLite database file used for traceability and analytics.
# In production this becomes a PostgreSQL URL (see docs/04_mlops.md).
DB_PATH = Path(_env("IVP_DB_PATH", str(PROJECT_ROOT / "artifacts" / "ivp.db")))


# ----------------------------------------------------------------------------
# Which MVTec LOCO category we inspect.
# Options: breakfast_box | juice_bottle | pushpins | screw_bag | splicing_connectors
# ----------------------------------------------------------------------------
CATEGORY = _env("IVP_CATEGORY", "screw_bag")


# ----------------------------------------------------------------------------
# Model selection.
# "auto"      -> use a trained model if present, else fall back to the dummy model
# "autoencoder" -> convolutional autoencoder anomaly detector (recommended)
# "classifier"  -> ResNet18 transfer-learning good/defect classifier
# "dummy"       -> numpy-only baseline (no PyTorch needed, used for tests/CI)
# ----------------------------------------------------------------------------
MODEL_BACKEND = _env("IVP_MODEL_BACKEND", "auto")

# Image side length fed to the model. 256 is a good balance of speed and detail.
IMAGE_SIZE = int(_env("IVP_IMAGE_SIZE", "256"))

# Anomaly decision threshold. If a model was trained, the calibrated value stored
# next to the weights wins; this is only the fallback when nothing is calibrated.
DEFAULT_THRESHOLD = float(_env("IVP_THRESHOLD", "0.5"))


# ----------------------------------------------------------------------------
# Service / runtime
# ----------------------------------------------------------------------------
API_HOST = _env("IVP_API_HOST", "0.0.0.0")
API_PORT = int(_env("IVP_API_PORT", "8000"))

# Line identifier written into every inspection record (multi-line / multi-factory).
LINE_ID = _env("IVP_LINE_ID", "LINE-01")
STATION_ID = _env("IVP_STATION_ID", "STATION-A")


def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet. Safe to call repeatedly."""
    for d in (MODELS_DIR, IMAGE_STORE_DIR, DB_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick way to inspect the active configuration: `python -m src.config`
    ensure_dirs()
    for key, value in sorted(globals().items()):
        if key.isupper():
            print(f"{key:18} = {value}")
