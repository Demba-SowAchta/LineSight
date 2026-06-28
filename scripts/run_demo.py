"""
run_demo.py -- prove the whole platform works in 10 seconds, with NO dataset
and NO PyTorch required.

What it does
------------
1. Generates a handful of synthetic "good" parts (smooth grey plates) and a few
   "defective" ones (same plate + a bright foreign patch = a missing/extra part).
2. Calibrates the numpy-only DummyDetector on the good samples.
3. Builds the full agent pipeline (Acquisition -> Inference -> Decision ->
   Storage -> Notification) via the Orchestrator.
4. Inspects every sample, archives evidence + heatmaps, and writes one
   traceability row per part into the SQLite database.
5. Prints a summary pulled straight back out of the database.

This is the fastest way to see the architecture end-to-end before you download
the real MVTec LOCO dataset and train a real model. Run it with:

    python scripts/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make `import src...` work whether you run this from the repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.inference_agent import InferenceAgent  # noqa: E402
from src.agents.orchestrator import Orchestrator  # noqa: E402
from src.database.db import init_db, summary_stats  # noqa: E402
from src.models.dummy_detector import DummyDetector  # noqa: E402

RNG = np.random.default_rng(42)
IMG = 256


def make_good() -> np.ndarray:
    """A clean part: uniform grey with mild sensor noise."""
    base = np.full((IMG, IMG, 3), 128, dtype=np.float32)
    base += RNG.normal(0, 4, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def make_defective() -> np.ndarray:
    """A faulty part: a clean plate with a bright foreign patch (wrong/extra component)."""
    img = make_good().astype(np.float32)
    y, x = RNG.integers(40, IMG - 80, size=2)
    img[y:y + 50, x:x + 50] += 110  # bright square = anomaly
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    print("=" * 64)
    print(" Industrial Vision Platform -- end-to-end demo (dummy backend)")
    print("=" * 64)

    init_db()

    # 1) Synthetic samples ------------------------------------------------
    good = [make_good() for _ in range(5)]
    bad = [make_defective() for _ in range(4)]

    # 2) Calibrate the numpy-only detector on GOOD parts only (anomaly-style)
    detector = DummyDetector().fit(good)

    # 3) Wire the pipeline, injecting our fitted detector into the Inference agent.
    #    Every other agent keeps its sensible default -- this single line is the
    #    "how to swap a model" mechanism in action.
    orch = Orchestrator(inference=InferenceAgent(detector=detector),
                        model_version="demo")

    # 4) Inspect everything ----------------------------------------------
    print("\nInspecting parts...\n")
    for i, img in enumerate(good):
        r = orch.inspect_one(part_id=f"GOOD_{i:02d}", image=img)
        print(f"  {r['part_id']}: {r['verdict']:4s}  score={r['score']:.3f}  "
              f"thr={r['threshold']:.3f}  {r['latency_ms']}ms")
    for i, img in enumerate(bad):
        r = orch.inspect_one(part_id=f"BAD_{i:02d}", image=img)
        print(f"  {r['part_id']}:  {r['verdict']:4s}  score={r['score']:.3f}  "
              f"thr={r['threshold']:.3f}  defect={r['defect_type']}  {r['latency_ms']}ms")

    # 5) Read the traceability summary back from the database -------------
    stats = summary_stats()
    print("\n" + "-" * 64)
    print(" Traceability summary (read from SQLite):")
    print(f"   total inspected : {stats['total']}")
    print(f"   passed          : {stats['passed']}")
    print(f"   failed          : {stats['failed']}")
    print(f"   pass rate       : {stats['pass_rate']:.1%}")
    print(f"   avg latency     : {stats['avg_latency_ms']:.2f} ms")
    print("-" * 64)
    print("\nDone. Evidence images + heatmaps were archived under the image store,")
    print("and one row per part was written to the database. Launch the dashboard")
    print("with `make app` to browse them.\n")


if __name__ == "__main__":
    main()
