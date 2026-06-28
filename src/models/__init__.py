"""Model package: swappable detectors behind one BaseDetector interface."""

from src.models.base import BaseDetector, DetectionResult
from src.models.factory import load_detector

__all__ = ["BaseDetector", "DetectionResult", "load_detector"]
