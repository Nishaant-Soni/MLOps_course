#!/usr/bin/env bash
# Delete the Cloud Run service and Artifact Registry repo to stop any charges.
# Usage: PROJECT_ID=my-project ./cleanup.sh
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID, e.g. PROJECT_ID=my-project ./cleanup.sh}"
REGION="${REGION:-us-east1}"
SERVICE="${SERVICE:-hello-world}"
REPO="${REPO:-cloud-runner-lab}"

gcloud run services delete "${SERVICE}" --region "${REGION}" --project "${PROJECT_ID}" --quiet
gcloud artifacts repositories delete "${REPO}" --location "${REGION}" --project "${PROJECT_ID}" --quiet
echo "Cleanup complete."
