#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="foodai-codex-temporal:latest"
CONTAINER_NAME="foodai-operational-refresh"

mkdir -p "${HOME}/.foodai-codex-home"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Running locked operational refresh in Docker..."

docker run --rm -it \
  --gpus all \
  --name "${CONTAINER_NAME}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/home/codex \
  -v "${HOME}/.foodai-codex-home:/home/codex" \
  -v "${REPO_ROOT}:/workspace/foodai" \
  -w /workspace/foodai \
  "${IMAGE}" \
  bash -lc 'python run_temporal_operational_refresh_v1.py --project-root /workspace/foodai'
