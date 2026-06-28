"""
Tests for the Industrial Vision Platform.

These run with NO PyTorch and NO dataset -- they use the numpy dummy model and
synthetic images, so CI stays fast and green. They prove the contracts that the
whole system depends on:
  - every detector obeys the BaseDetector interface
  - the agent pipeline produces a verdict and writes a traceable DB record
  - the database aggregates KPIs correctly

RUN:  pytest -q
"""

from __future__ import annotations

import numpy as np

from src.database import db
from src.models.dummy_detector import DummyDetector
from src.models.base import DetectionResult


def _good_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (np.ones((256, 256, 3)) * 120 + rng.normal(0, 5, (256, 256, 3))).clip(0, 255).astype("uint8")


def _bad_image(seed: int = 0) -> np.ndarray:
    img = _good_image(seed)
    img[90:150, 90:150] = 250  # bright defect patch
    return img


def test_detector_interface_contract():
    det = DummyDetector()
    res = det.predict(_good_image())
    assert isinstance(res, DetectionResult)
    assert 0.0 <= res.score <= 1.0
    assert isinstance(res.is_anomaly, bool)
    assert 0.0 <= res.confidence <= 1.0


def test_dummy_separates_good_from_bad():
    det = DummyDetector()
    good_score = det.predict(_good_image()).score
    bad_score = det.predict(_bad_image()).score
    # The defective image must score clearly higher than the good one.
    assert bad_score > good_score + 0.2


def test_pipeline_writes_traceable_record(tmp_path, monkeypatch):
    # Point the DB and image store at a temp folder so tests stay isolated.
    monkeypatch.setattr("src.config.DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr("src.config.IMAGE_STORE_DIR", tmp_path / "imgs")
    monkeypatch.setattr("src.config.MODEL_BACKEND", "dummy")

    # Import after patching so agents pick up the patched config.
    from src.agents import Orchestrator
    orch = Orchestrator(model_version="test")

    good = orch.inspect_one("good-1", _good_image())
    bad = orch.inspect_one("bad-1", _bad_image())

    assert good["verdict"] == "PASS"
    assert bad["verdict"] == "FAIL"
    assert bad["inspection_id"] != good["inspection_id"]

    stats = db.summary_stats(db_path=tmp_path / "t.db")
    assert stats["total"] == 2
    assert stats["failed"] == 1
    assert stats["passed"] == 1


def test_summary_stats_empty(tmp_path):
    db.init_db(db_path=tmp_path / "empty.db")
    stats = db.summary_stats(db_path=tmp_path / "empty.db")
    assert stats["total"] == 0
    assert stats["pass_rate"] == 0.0
