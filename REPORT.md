# Visual Detection of Product Assembly Errors
### Computer Vision for Industry 5.0 — Project #3

**Team:** Sow Achta Demba, David Gwodog, Dhia Rekik
**Dataset:** MVTec LOCO AD (Logical and Structural Anomaly Detection)

---

## 1. Problem & approach

Manufacturing lines need to catch assembly defects — missing parts, wrong
placement, damage, packaging errors — before products reach customers. Manual
inspection is slow, costly, and inconsistent. We built an automated visual
inspection system that classifies an image of an assembled product as **good**
or **defective**, localizes the defect, and records every decision for
traceability.

We frame this as **anomaly detection** rather than ordinary classification. In a
real factory, good units are abundant but defects are rare, varied, and
unpredictable — so we train on *good parts only* and flag anything that deviates.
This matches the MVTec LOCO dataset and the realities of production.

## 2. Dataset exploration

MVTec LOCO AD contains five product categories (breakfast_box, juice_bottle,
pushpins, screw_bag, splicing_connectors). Each category provides:

- `train/good` — good images only (training set),
- `validation/good` — good images (for threshold calibration),
- `test/` — `good`, plus `logical_anomalies` and `structural_anomalies`.

The dataset distinguishes two defect families, which shaped our decision logic:

- **Structural anomalies** — a localized physical fault (damage, misplacement);
  these produce a small, concentrated hot region in an error map.
- **Logical anomalies** — the part is locally fine but globally wrong (missing
  item, wrong count/combination); these are diffuse and harder.

We work primarily with the **screw_bag** category. Training uses only good
images; the test anomalies are reserved strictly for evaluation, with no leakage
into training.

## 3. Model training

We implemented three interchangeable detectors behind a common interface, so the
rest of the system never changes when we swap models:

1. **Baseline (numpy)** — a non-neural detector scoring brightness/edge/local
   block deviation versus calibrated good parts. It exists so the full system
   runs with zero heavy dependencies (useful for development and demos).
2. **Convolutional autoencoder (our main model)** — trained to reconstruct good
   images at 256×256. The reconstruction error is the anomaly score, and the
   per-pixel error map is the defect heatmap. We chose this as the primary model
   because it trains on good images only and gives natural defect localization.
3. **ResNet18 transfer learning (alternative)** — a pretrained backbone (frozen)
   with a new good/defect head, as a fast classical baseline.

**Training procedure (autoencoder):** train the encoder–decoder on `train/good`
to minimize reconstruction error; then **calibrate the PASS/FAIL threshold** by
running the trained model over `validation/good` and taking a high percentile
(95–99th) of those reconstruction errors. A good part should reconstruct well
(low error, below threshold); an anomaly reconstructs poorly (high error, above
threshold). The threshold is saved *with* the model so inference is reproducible.

**Switching models** is a single environment variable (`IVP_MODEL_BACKEND`),
which is what makes comparing the three approaches straightforward.

## 4. Evaluation

We evaluate on the held-out test split (good + logical + structural anomalies)
and report, computed without external ML libraries:

- **Confusion matrix** (TP/FP/TN/FN),
- **Precision, Recall, F1**,
- **Accuracy**,
- **AUROC** (threshold-independent separability, via rank-sum).

**Recall is our primary metric.** A false negative — a defect passed as good —
reaches the customer and is the expensive error; a false positive only costs a
re-check. We therefore tune the threshold to favour recall, accepting a few more
false alarms, with the exact trade-off set to the line's tolerance. Logical
anomalies are expected to be the hardest cases for a reconstruction model, which
is consistent with the dataset's design and a useful point of discussion.

*(Concrete numbers are produced by `make evaluate` once the model is trained on a
machine with PyTorch — e.g. Google Colab — and are inserted here from the
generated confusion matrix and ROC artifacts.)*

## 5. Prototype application

The prototype is a **Streamlit** dashboard with three tabs:

- **Inspect** — upload an image, get the verdict, the defect heatmap overlay,
  the score vs threshold, and the latency.
- **Dashboard** — KPIs (total inspected, pass rate, average latency), the recent
  inspections table, and a defect-type breakdown chart — all read from the
  traceability database.
- **About** — how to switch the model and category.

A **FastAPI** service exposes the same pipeline (`/inspect`, `/stats`,
`/health`) for machine-to-machine use, and a webcam script provides a live demo.
Every inspection writes one row to a **SQLite** database (timestamp,
line/station, verdict, defect type, score, model version, latency, image paths),
giving full traceability — the foundation for analytics and audits.

## 6. System design (why it's structured this way)

The pipeline is split into five **agents** with one job each — Acquisition,
Inference, Decision, Storage, Notification — coordinated by an Orchestrator. This
isolation means the camera source, the model, the quality rules, the storage
backend, and the alert channel can each change without touching the others, and
each maps cleanly onto a future microservice. The same design lets the project
scale from this single-station prototype toward a production platform
(Kubernetes, MLflow, monitoring, PLC/MES integration), documented in the
accompanying `docs/`.

## 7. Conclusion & future work

We delivered a working, traceable, demoable inspection system with swappable
anomaly-detection models and a clear path to production. Next steps: train and
report final metrics per category on Colab, add drift monitoring to trigger
retraining, and integrate with a PLC via MQTT/OPC-UA for closed-loop rejection.

---

*Repository: full code, the runnable demo (`python scripts/run_demo.py`), and the
architecture documentation in `docs/`.*
