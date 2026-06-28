# deploy/

Placeholder infrastructure stubs for the **scale-up path** (see
`docs/05_devops_cicd.md`). These are intentionally minimal — they mark where
production infrastructure-as-code lives, not full production manifests.

- `helm/` — Helm chart skeleton for deploying the API + dashboard to Kubernetes.
- `terraform/` — Terraform skeleton for provisioning cluster, database, and
  object storage.

The system runs fully today via `docker compose up` (repo root). Reach for these
only when moving to a multi-node Kubernetes deployment.
