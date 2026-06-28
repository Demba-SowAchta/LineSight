"""
AGENT 2 of 6 -- InferenceAgent

ROLE:   Turn an image into a raw anomaly result by running the model. It is the
        only agent that touches a model. It measures its own latency (for
        monitoring) and stays completely model-agnostic.

WHY A SEPARATE AGENT: the model is the part most likely to change -- you retrain
it weekly, you A/B test versions, you swap autoencoder for classifier. Because
this agent only depends on the BaseDetector *interface*, swapping the model is a
config change (IVP_MODEL_BACKEND), not a code change. See src/models/factory.py.

INTERACTS WITH: receives a frame from the Orchestrator, returns a DetectionResult
(score + heatmap) to the DecisionAgent.
"""

from __future__ import annotations

import time

import numpy as np

from src.models import BaseDetector, DetectionResult, load_detector


class InferenceAgent:
    name = "inference"

    def __init__(self, detector: BaseDetector | None = None):
        # If no detector is injected, ask the factory for the configured one.
        # Injecting one in tests is how we keep this agent easy to unit-test.
        self.detector = detector or load_detector()

    @property
    def model_name(self) -> str:
        return self.detector.name

    def run(self, image: np.ndarray) -> tuple[DetectionResult, float]:
        """
        Run the model on one image.
        Returns the DetectionResult and the inference latency in milliseconds.
        """
        start = time.perf_counter()
        result = self.detector.predict(image)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return result, latency_ms
