# 00 — Executive Summary

## What this project is

An **automated visual inspection platform** that looks at a photo of an
assembled product and decides, in real time, whether it is **good** or
**defective** — then records that decision so it can be traced later.

It was built around a concrete course project (detecting assembly errors on the
**MVTec LOCO AD** dataset) but is structured the way a real production system
would be, so the same code can grow from a laptop demo to a factory deployment.

## The honest scope (read this first)

There are two layers in this repository, and it is important not to confuse
them:

1. **What actually runs today — a real, working system.**
   A complete inspection pipeline you can run right now: it scores an image,
   decides PASS/FAIL, draws a defect heatmap, archives the evidence, writes a
   traceability record to a database, raises an alert on failures, and serves
   all of this through both a REST API and a web dashboard. It runs on a plain
   machine with no GPU, and even with *no dataset and no PyTorch* thanks to a
   built-in baseline model. This is genuinely deployable for a single station.

2. **What is documented as the scale-up path — the enterprise design.**
   The original brief asked for an industrial platform: Kubernetes, MLflow,
   Prometheus/Grafana, Vault, PLC/MES/SCADA integration, multi-factory scaling,
   and so on. Standing all of that infrastructure up is months of work and a
   real budget; it cannot be "delivered" as runnable code in a student project.
   So it is delivered as **architecture and engineering decisions** (docs
   `01`–`13`), with the working system above as the seed that those layers wrap
   around. Every diagram explains *where* a piece would plug into the code that
   already exists.

This split is deliberate. A demo that pretends to be a factory platform helps
no one; a factory platform with no running core is just slides. You get a
running core **and** the blueprint to grow it.

## Business value (why a factory would want this)

- **Stop defects reaching customers** — every part is checked, not a sample.
- **Cut manual inspection cost** — operators handle exceptions, not every unit.
- **Traceability & analytics** — every decision is stored with its score,
  model version, timestamp and station, so you can audit and improve.
- **Continuous operation** — stateless inference services that can be
  replicated and restarted without losing data.

## Key design choices at a glance

| Decision | Why |
|---|---|
| **Anomaly detection** (train on good only) | Factories have many good units, few/unknown defects. |
| **Agent-based pipeline** | Each step (acquire, infer, decide, store, notify) is isolated and swappable. |
| **Swappable model backend** | Switch dummy → autoencoder → classifier with one env var. |
| **SQLite now, Postgres later** | Zero-config for the demo; documented one-step migration for production. |
| **Runs without GPU/torch** | The demo never blocks on heavy dependencies. |

## Where to go next

- New to the project? Read the main `README.md`, then `docs/09_agents.md`.
- Want the big picture? `docs/01_architecture.md`.
- Want to ship it for real? `docs/10_production_checklist.md` and
  `docs/13_roadmap.md`.
