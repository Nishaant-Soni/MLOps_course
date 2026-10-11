#!/usr/bin/env bash
# Build, push (Artifact Registry) and deploy to Cloud Run.
# Usage: PROJECT_ID=my-project ./deploy.sh
set -euo pipefail

# Always build from this script's folder, wherever it is launched from.
cd "$(dirname "${BASH_SOURCE[0]}")"

: "${PROJECT_ID:?Set PROJECT_ID, e.g. PROJECT_ID=my-project ./deploy.sh}"
REGION="${REGION:-us-east1}"
SERVICE="${SERVICE:-hello-world}"
REPO="${REPO:-cloud-runner-lab}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"
CONCURRENCY="${CONCURRENCY:-10}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:$(date +%Y%m%d%H%M%S)"

gcloud config set project "${PROJECT_ID}"

echo ">> Enabling APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

echo ">> Ensuring Artifact Registry repo '${REPO}' exists"
gcloud artifacts repositories describe "${REPO}" --location "${REGION}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${REPO}" --repository-format docker --location "${REGION}"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo ">> Building image (linux/amd64)"
docker build --platform linux/amd64 -t "${IMAGE}" .

echo ">> Pushing ${IMAGE}"
docker push "${IMAGE}"

echo ">> Deploying to Cloud Run"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --allow-unauthenticated \
  --min-instances "${MIN_INSTANCES}" \
  --max-instances "${MAX_INSTANCES}" \
  --concurrency "${CONCURRENCY}" \
  --memory 256Mi \
  --cpu 1 \
  --set-env-vars "GREETING=${GREETING:-Hello from Cloud Run!}"

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format 'value(status.url)')"
echo
echo "Deployed: ${URL}"
echo "Try:      curl ${URL}/health"
