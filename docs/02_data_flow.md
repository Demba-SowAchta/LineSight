# 02 — Data Flow

This traces a single part from camera to stored verdict, then describes the
dataset flow used for training.

## Inference data flow (one part)

```mermaid
sequenceDiagram
    participant Cam as Camera/Upload
    participant Acq as AcquisitionAgent
    participant Inf as InferenceAgent
    participant Mdl as Detector (model)
    participant Dec as DecisionAgent
    participant Sto as StorageAgent
    participant DB as Database
    participant Not as NotificationAgent

    Cam->>Acq: raw frame (bytes / array)
    Acq->>Inf: (part_id, image ndarray)
    Inf->>Mdl: predict(image)
    Mdl-->>Inf: DetectionResult(score, heatmap)
    Inf-->>Dec: result + latency_ms
    Dec->>Dec: score vs threshold -> verdict
    Dec->>Dec: heatmap hot_fraction -> structural/logical
    Dec-->>Sto: decision dict
    Sto->>DB: INSERT inspection row
    Sto->>Sto: save evidence + heatmap overlay
    Dec-->>Not: decision dict
    Not->>Not: if FAIL -> emit alert (console/MQTT)
```

### What each stage produces

| Stage | Input | Output |
|---|---|---|
| Acquisition | camera/folder/upload | `(part_id, image)` as a numpy array |
| Inference | image | `score`, `heatmap`, `latency_ms` |
| Decision | score + heatmap | `verdict`, `defect_type`, `severity`, `confidence` |
| Storage | decision + image | DB row id, archived image + heatmap paths |
| Notification | decision | alert (only when FAIL) |

### The traceability record

Every inspection writes one row with: timestamp, line/station, category,
part id, verdict, defect type, score, threshold, confidence, model name +
version, latency, and the paths to the archived image and heatmap. That row is
what makes the system **auditable**: you can answer "why was part X failed, by
which model, when?" months later.

## Dataset / training data flow

```mermaid
flowchart LR
    RAW[MVTec LOCO folders<br/>train/good, validation/good, test/*] --> PREP[prepare_data.py<br/>build manifest.csv]
    PREP --> TRAIN[train_anomaly.py<br/>fit on GOOD only]
    TRAIN --> CAL[calibrate threshold<br/>from good-val errors]
    CAL --> MODEL[(saved model<br/>weights + calibration json)]
    MODEL --> EVAL[evaluate.py<br/>confusion, P/R/F1, AUROC]
    MODEL --> FACTORY[factory.load_detector<br/>used at inference]
```

Key point: the model is trained on **good images only**. The test anomalies are
used solely for *evaluation*, never for training — this is what makes it an
anomaly-detection setup rather than ordinary classification.

See `docs/03_ai_pipeline.md` for the modelling detail and `docs/04_mlops.md` for
how versioning/registry would wrap this flow in production.
