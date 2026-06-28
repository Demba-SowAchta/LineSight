# Terraform skeleton (scale-up stub). Not production-complete.
# Provisions the platform's infrastructure: cluster, database, object storage.
# Fill in a real provider (aws/gcp/azure) and backend before use.

terraform {
  required_version = ">= 1.5"
  # backend "s3" { ... }   # store state remotely in production
}

variable "environment" {
  description = "Deployment environment (dev/test/staging/prod)"
  type        = string
  default     = "dev"
}

# Placeholders — replace with real resources for your cloud:
# - kubernetes_cluster (with a GPU node pool for training/batch)
# - managed_postgresql (traceability DB)
# - object_storage_bucket (image/evidence store)
# - container_registry (built images)

output "note" {
  value = "Stub only. See docs/05_devops_cicd.md for the intended topology."
}
