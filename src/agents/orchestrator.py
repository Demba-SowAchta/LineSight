"""
AGENT 6 of 6 -- Orchestrator (the conductor)

ROLE:   Wire the five worker agents into one inspection flow and run them in order.
        It owns the "story" of an inspection; the workers own the "how" of each step.

THE FLOW (one part through the line):
    Acquisition -> Inference -> Decision -> Storage -> Notification
        pixels       score        verdict     record      alert

WHY A SEPARATE AGENT: the orchestrator is the single place that knows the order of
steps. Want to add a step (e.g. a PreprocessingAgent, or a second model for
A/B testing)? You edit ONLY this file. The workers stay unaware of each other,
which is what keeps the system easy to reason about and to extend.

HOW TO SWAP AGENTS: every worker is injected in __init__. To use a different
source, model, or alert channel, pass a different agent instance here -- no other
file changes. That is the whole point of the agent design.

DEPLOYMENT NOTE: in the demo this runs as ONE process (a "monolith"), which is
perfect for a laptop. In production each agent becomes its own microservice that
communicates over a queue (Kafka/RabbitMQ). The code boundaries are already drawn
along the seams where you would cut for microservices -- see docs/01_architecture.md.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from src.agents.acquisition_agent import AcquisitionAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.inference_agent import InferenceAgent
from src.agents.notification_agent import NotificationAgent
from src.agents.storage_agent import StorageAgent


class Orchestrator:
    def __init__(self,
                 acquisition: AcquisitionAgent | None = None,
                 inference: InferenceAgent | None = None,
                 decision: DecisionAgent | None = None,
                 storage: StorageAgent | None = None,
                 notification: NotificationAgent | None = None,
                 model_version: str | None = None):
        # Each agent defaults to a sensible instance but can be replaced.
        self.acquisition = acquisition or AcquisitionAgent()
        self.inference = inference or InferenceAgent()
        self.decision = decision or DecisionAgent()
        self.storage = storage or StorageAgent()
        self.notification = notification or NotificationAgent()
        self.model_version = model_version

    def inspect_one(self, part_id: str, image: np.ndarray) -> dict[str, Any]:
        """
        Run the full pipeline on a single image and return a summary dict.
        This is the function the API and the Streamlit app both call.
        """
        # 1) INFERENCE -- model score + heatmap (acquisition already happened upstream)
        result, latency_ms = self.inference.run(image)

        # 2) DECISION -- raw score -> PASS/FAIL + defect type + severity
        decision = self.decision.decide(result)

        # 3) STORAGE -- archive evidence + write the traceability record
        inspection_id = self.storage.save(
            part_id=part_id,
            image=image,
            decision=decision,
            model_name=self.inference.model_name,
            model_version=self.model_version,
            latency_ms=latency_ms,
            heatmap=result.heatmap,
        )

        # 4) NOTIFICATION -- alert the floor on failures only
        self.notification.notify(inspection_id=inspection_id, part_id=part_id, decision=decision)

        # Return everything the caller (UI/API) needs to display the result.
        return {
            "inspection_id": inspection_id,
            "part_id": part_id,
            "verdict": decision["verdict"],
            "defect_type": decision["defect_type"],
            "severity": decision["severity"],
            "score": decision["score"],
            "threshold": decision["threshold"],
            "confidence": decision["confidence"],
            "latency_ms": round(latency_ms, 2),
            "model_name": self.inference.model_name,
            "heatmap": result.heatmap,  # numpy array (or None) for the UI to draw
        }

    def inspect_folder(self, folder: str) -> Iterator[dict[str, Any]]:
        """Stream results for every image in a folder -> batch inspection."""
        for part_id, image in self.acquisition.from_folder(folder):
            yield self.inspect_one(part_id, image)
