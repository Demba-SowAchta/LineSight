"""
FastAPI inference service -- the production-facing REST API.

It wraps the SAME Orchestrator the demo uses, so the API and the app always agree.
Three endpoints:
    GET  /health   -> liveness/readiness probe (used by Docker/Kubernetes)
    GET  /stats    -> live KPIs from the database (for monitoring/dashboards)
    POST /inspect  -> upload one image, get back the verdict JSON

WHY FastAPI: it is fast, async, and auto-generates interactive docs at /docs, so
operators and integrators can try the API from a browser with no extra tooling.

RUN:  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
      then open http://localhost:8000/docs
"""

from __future__ import annotations

import io

import numpy as np

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the API deps:  pip install fastapi uvicorn python-multipart") from exc

from PIL import Image

from src import config
from src.agents import Orchestrator
from src.database import db

app = FastAPI(
    title="Industrial Vision Platform - Inspection API",
    version="1.0.0",
    description="Real-time assembly-error detection for manufacturing lines.",
)

# Build the orchestrator ONCE at startup so the model is loaded a single time,
# not on every request. This is the difference between 5 ms and 5 s per call.
orchestrator: Orchestrator | None = None


@app.on_event("startup")
def _startup() -> None:
    global orchestrator
    config.ensure_dirs()
    db.init_db()
    orchestrator = Orchestrator(model_version=config.MODEL_BACKEND)


@app.get("/health")
def health() -> dict:
    """Cheap check that the service is up and a model is loaded."""
    ready = orchestrator is not None
    return {
        "status": "ok" if ready else "starting",
        "model": orchestrator.inference.model_name if ready else None,
        "line_id": config.LINE_ID,
        "category": config.CATEGORY,
    }


@app.get("/stats")
def stats() -> dict:
    """Live KPIs straight from the traceability database."""
    return db.summary_stats()


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), part_id: str = "api-upload") -> JSONResponse:
    """
    Inspect one uploaded image. Returns the verdict, defect type, score, and the
    database id of the stored record (so the part stays traceable).
    """
    raw = await file.read()
    image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)

    result = orchestrator.inspect_one(part_id, image)
    # The heatmap is a numpy array; drop it from the JSON (binary, not JSON-friendly).
    result.pop("heatmap", None)
    return JSONResponse(result)
