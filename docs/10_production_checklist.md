# 10 — Production Readiness Checklist

A go/no-go list before putting this on a real line. Items marked ✅ are satisfied
by what's in the repo today; ☐ are the scale-up work described in the other docs.

## Infrastructure readiness
- ✅ Containerized, runs as non-root, healthcheck present
- ✅ Stateless inference (state lives in DB + image store, not in the process)
- ☐ Kubernetes manifests / Helm chart finalized (`deploy/`)
- ☐ Terraform for cluster, managed DB, object store, GPU pool
- ☐ Autoscaling (HPA) configured and load-tested

## AI model readiness
- ✅ Reproducible training scripts + calibration saved with the model
- ✅ Evaluation with confusion matrix, precision/recall/F1, AUROC
- ☐ Model meets the promotion gate (e.g. recall ≥ 0.95) on a frozen holdout
- ☐ Shadow-tested against current production model
- ☐ Per-category models trained and registered

## Data readiness
- ✅ Clear dataset layout + download guide (`scripts/download_data.md`)
- ✅ Train-on-good-only discipline (no test leakage)
- ☐ Data versioned (DVC) and quality-validated
- ☐ Retention + privacy policy for stored images

## Security readiness
- ✅ No secrets in code; secrets git-ignored; minimal base image
- ☐ Keycloak SSO + RBAC roles enforced
- ☐ Vault for runtime secrets
- ☐ TLS/mTLS; encrypted volumes
- ☐ Trivy scan passing in CI; Falco runtime monitoring

## Monitoring readiness
- ✅ Per-inspection latency/score/verdict captured; stats endpoint + dashboard
- ☐ Prometheus + Grafana dashboards live
- ☐ Drift detection wired with retraining trigger
- ☐ Alertmanager routes to on-call / line lead
- ☐ SLOs defined with error budgets

## Operational readiness
- ✅ One-command demo; Makefile targets; README runbook basics
- ☐ Runbooks for common incidents (model down, DB full, drift alarm)
- ☐ On-call rotation and escalation defined
- ☐ Capacity plan per line (throughput vs SLO)

## Manufacturing integration readiness
- ✅ Notification hook with MQTT channel (bridge stub)
- ☐ OPC-UA / MQTT integration tested with the actual PLC
- ☐ Fail-safe behavior agreed with quality/safety (what the line does if vision
  is unavailable)
- ☐ Heartbeat + backpressure validated

## Disaster recovery readiness
- ☐ Automated DB + image-store backups, restore tested
- ☐ Defined RTO/RPO
- ☐ Multi-zone / failover for the inference service
- ☐ Documented rollback (model and app) rehearsed

## Go-live gate
Ship to a single line only when every **Security**, **AI model**, and
**Manufacturing integration** item is ✅ for that line, plus backups tested.
Everything else can follow a phased rollout (see `docs/13_roadmap.md`).
