"""
AGENT 3 of 6 -- DecisionAgent

ROLE:   Convert the model's raw score into a BUSINESS decision: PASS or FAIL, plus
        a human-readable defect category. This is where engineering meets the
        factory's quality policy. Keeping it separate means quality engineers can
        tune rules (thresholds, severity, defect naming) WITHOUT touching the model.

WHY A SEPARATE AGENT: the model outputs a number; the business needs a verdict and
a reason. Those are different concerns. A future rule like "two borderline parts
in a row = stop the line" lives here, not in the model.

INTERACTS WITH: receives a DetectionResult from the InferenceAgent, emits a
Decision dict consumed by the StorageAgent and NotificationAgent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models import DetectionResult


class DecisionAgent:
    name = "decision"

    def __init__(self, hot_fraction_for_structural: float = 0.015):
        # If more than this fraction of the heatmap is "hot", we call the defect
        # structural (a localised physical problem) rather than logical.
        self.hot_fraction = hot_fraction_for_structural

    def decide(self, result: DetectionResult) -> dict[str, Any]:
        """Return a decision dictionary ready to be stored and acted upon."""
        verdict = "FAIL" if result.is_anomaly else "PASS"
        defect_type = self._classify_defect(result) if result.is_anomaly else "good"

        return {
            "verdict": verdict,
            "defect_type": defect_type,
            "score": result.score,
            "threshold": result.threshold,
            "confidence": result.confidence,
            "severity": self._severity(result),
        }

    def _classify_defect(self, result: DetectionResult) -> str:
        """
        A simple, explainable heuristic that names the defect family.

        MVTec LOCO defines two families:
          - structural: a localised physical defect (scratch, missing screw) ->
            shows up as a small bright blob in the heatmap.
          - logical:    a high-level rule is broken (wrong count, wrong combo) ->
            the image looks globally "off" with no single hot blob.
        We infer which by how concentrated the heatmap is.
        """
        if result.heatmap is None:
            return "defect"  # plain classifier: no map, leave it generic
        heat = np.asarray(result.heatmap)
        hot_fraction = float((heat > 0.6).mean())  # share of strongly-hot pixels
        return "structural" if hot_fraction >= self.hot_fraction else "logical"

    @staticmethod
    def _severity(result: DetectionResult) -> str:
        """Map confidence onto an operator-friendly severity label."""
        if not result.is_anomaly:
            return "none"
        if result.confidence >= 0.6:
            return "high"
        if result.confidence >= 0.3:
            return "medium"
        return "low"
