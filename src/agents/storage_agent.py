"""
AGENT 4 of 6 -- StorageAgent

ROLE:   Make every inspection PERMANENT and AUDITABLE. It saves the evidence image
        (and heatmap) to disk and writes one row to the database with the full
        context: when, which line/station, verdict, defect type, score, model
        version, latency, and the image paths. This is the "traceability" pillar.

WHY A SEPARATE AGENT: persistence is a cross-cutting concern. Isolating it means
we can later send the same record to PostgreSQL, S3, or a data lake by editing
ONE agent, while the pipeline keeps calling `storage.save(...)` unchanged.

INTERACTS WITH: receives the image + the DecisionAgent's verdict + run metadata,
writes to src/database/db.py and the image store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src import config
from src.database import db
from src.utils.images import overlay_heatmap, save_image


class StorageAgent:
    name = "storage"

    def __init__(self, image_store: Path | None = None):
        self.image_store = Path(image_store or config.IMAGE_STORE_DIR)
        db.init_db()  # make sure the table exists before the first write

    def save(self, *, part_id: str, image: np.ndarray, decision: dict[str, Any],
             model_name: str, model_version: str | None, latency_ms: float,
             heatmap: np.ndarray | None = None) -> int:
        """
        Archive the evidence and insert a traceability record.
        Returns the new inspection id (a handle the operator can quote later).
        """
        # Folder layout groups evidence by day and verdict -> easy to audit.
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        verdict = decision["verdict"]
        out_dir = self.image_store / day / verdict
        stem = f"{config.LINE_ID}_{part_id}_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"

        image_path = save_image(image, out_dir / f"{stem}.jpg")

        heatmap_path = None
        if heatmap is not None:
            overlay = overlay_heatmap(image, heatmap)
            heatmap_path = save_image(overlay, out_dir / f"{stem}_heatmap.jpg")

        record = {
            "line_id": config.LINE_ID,
            "station_id": config.STATION_ID,
            "category": config.CATEGORY,
            "part_id": part_id,
            "verdict": verdict,
            "defect_type": decision.get("defect_type"),
            "score": decision["score"],
            "threshold": decision["threshold"],
            "confidence": decision.get("confidence"),
            "model_name": model_name,
            "model_version": model_version,
            "latency_ms": latency_ms,
            "image_path": str(image_path),
            "heatmap_path": str(heatmap_path) if heatmap_path else None,
        }
        return db.insert_inspection(record)
