"""
Agents package -- the six cooperating components of the inspection pipeline.

    AcquisitionAgent  -> gets pixels in
    InferenceAgent    -> runs the (swappable) model
    DecisionAgent     -> score -> PASS/FAIL + defect type
    StorageAgent      -> evidence + traceability record
    NotificationAgent -> real-time alerts to operators / PLC
    Orchestrator      -> wires them together in order

See docs/09_agents.md for the full beginner->expert walkthrough.
"""

from src.agents.acquisition_agent import AcquisitionAgent
from src.agents.inference_agent import InferenceAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.storage_agent import StorageAgent
from src.agents.notification_agent import NotificationAgent
from src.agents.orchestrator import Orchestrator

__all__ = [
    "AcquisitionAgent",
    "InferenceAgent",
    "DecisionAgent",
    "StorageAgent",
    "NotificationAgent",
    "Orchestrator",
]
