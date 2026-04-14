#!/usr/bin/env bash
set -euo pipefail

cd /workspace/foodai

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [ "$#" -eq 0 ]; then
  set -- bash
fi

exec "$@"
