# 01 — Architecture

This document gives the **high-level** picture first, then the **detailed**
component view, then the **production / scale-up** architecture. The first one
is what runs in this repo; the last one is the documented target.

---

## 1. High-level architecture (what runs today)

A single inspection station: an image comes in, a verdict and a stored record
come out.

```mermaid
flowchart LR
    CAM[Camera / folder / upload] --> ACQ[Acquisition Agent]
    ACQ --> INF[Inference Agent<br/>swappable model]
    INF --> DEC[Decision Agent<br/>PASS / FAIL + defect type]
    DEC --> STO[Storage Agent]
    DEC --> NOT[Notification Agent]
    STO --> DB[(SQLite DB<br/>traceability)]
    STO --> FS[/Image store<br/>evidence + heatmaps/]
    NOT --> OPS[Operator / PLC alert]
    DB --> UI[Streamlit dashboard]
    DB --> API[FastAPI /stats]
```

The orchestrator wires these five agents together. The UI and API are just two
front doors onto the same `Orchestrator.inspect_one()` call.

---

## 2. Detailed component view

```mermaid
flowchart TB
    subgraph Frontends
      ST[Streamlit app<br/>src/app]
      FA[FastAPI service<br/>src/api]
      WC[Webcam demo<br/>scripts/webcam_demo.py]
    end

    subgraph Core["Agent pipeline (src/agents)"]
      ORCH[Orchestrator]
      A1[AcquisitionAgent]
      A2[InferenceAgent]
      A3[DecisionAgent]
      A4[StorageAgent]
      A5[NotificationAgent]
      ORCH --> A1 & A2 & A3 & A4 & A5
    end

    subgraph Models["Model layer (src/models)"]
      FAC[factory.load_detector]
      M0[DummyDetector<br/>numpy only]
      M1[AutoencoderDetector<br/>PyTorch]
      M2[ClassifierDetector<br/>ResNet18]
      FAC --> M0 & M1 & M2
    end

    subgraph Data["Persistence"]
      DB[(SQLite / Postgres)]
      IS[/Image store/]
    end

    ST --> ORCH
    FA --> ORCH
    WC --> ORCH
    A2 --> FAC
    A4 --> DB
    A4 --> IS
```

**The two seams that make this flexible:**

- `factory.load_detector(backend)` — the **only** place a model is chosen. The
  Inference agent never names a concrete model.
- The `Orchestrator(__init__)` — every agent is injected, so you can replace any
  one (a different camera source, a different alert channel) without touching
  the others.

See `docs/09_agents.md` for a line-by-line walkthrough.

---

## 3. Production / scale-up architecture (the documented target)

This is **not** in the repo as running infrastructure; it is the blueprint the
running core is designed to slot into. Each block notes which existing piece it
wraps.

```mermaid
flowchart TB
    subgraph Edge["Factory floor (edge)"]
      C1[Industrial cameras<br/>GigE / USB3] --> EG[Edge gateway<br/>runs AcquisitionAgent]
      EG --> EI[Edge inference<br/>InferenceAgent + ONNX/TensorRT]
      EI --> PLC[PLC / MES via OPC-UA / MQTT<br/>NotificationAgent channel]
    end

    subgraph Cluster["On-prem / cloud Kubernetes"]
      ING[Ingress + API gateway] --> SVC[Inference API<br/>FastAPI, replicated]
      SVC --> Q[(Queue: Kafka/RabbitMQ)]
      Q --> W[Batch workers / GPU pool]
      SVC --> PG[(PostgreSQL<br/>traceability)]
      SVC --> OBJ[/Object store<br/>images, S3-compatible/]
      MLF[MLflow registry] --> SVC
    end

    subgraph Ops["MLOps / Observability / Security"]
      PROM[Prometheus + Grafana]
      LOKI[Loki / ELK logs]
      VAULT[Vault secrets]
      CICD[CI/CD: GitHub Actions]
    end

    EI -. results .-> ING
    SVC --> PROM
    W --> PROM
    CICD --> SVC
    VAULT --> SVC
```

Mapping to detailed docs:

| Layer | Doc |
|---|---|
| Cameras, lighting, acquisition | `02_data_flow.md`, `03_ai_pipeline.md` |
| Model training, registry, versioning | `04_mlops.md` |
| Containers, K8s, Helm, Terraform | `05_devops_cicd.md` |
| CI/CD pipeline stages | `05_devops_cicd.md` |
| Metrics, logs, drift | `06_monitoring.md` |
| Auth, RBAC, secrets, scanning | `07_security.md` |
| PLC / MES / SCADA / OPC-UA / MQTT | `08_manufacturing_integration.md` |

---

## Why an agent-based design (the central decision)

A naive script would do `acquire → infer → decide → store` in one function. It
works until you need to: change the camera, swap the model, add a second alert
channel, run inference on a GPU box while storage stays local, or test one step
in isolation. Each of those becomes a risky edit to a tangled function.

By splitting the flow into **agents with one job each**, connected through a
thin orchestrator:

- you can replace any agent without touching the rest (open/closed principle);
- you can test each agent alone;
- the same boundaries become **service boundaries** when you later split the
  monolith into microservices (the agents map 1:1 onto future deployable units).

The monolith-now / microservices-later path is covered in `docs/13_roadmap.md`.
