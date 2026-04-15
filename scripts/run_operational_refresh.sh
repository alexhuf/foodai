#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

REFRESH_NAME="simple_loss_daysweeks_v2_operational_refresh_v1"
REPORT_DIR="${PROJECT_ROOT}/reports/backtests/temporal_multires/${REFRESH_NAME}"
LATEST_SUMMARY="${REPORT_DIR}/latest_case_summary.md"
FIRST_READ="${REPORT_DIR}/summary.md"

cd "${PROJECT_ROOT}"

echo "Running locked operational refresh..."
"${PYTHON_BIN}" run_temporal_operational_refresh_v1.py --project-root "${PROJECT_ROOT}"

echo
echo "Latest case summary:"
if [[ -f "${LATEST_SUMMARY}" ]]; then
  sed -n '1,80p' "${LATEST_SUMMARY}"
else
  echo "Missing expected latest summary: ${LATEST_SUMMARY}" >&2
  exit 1
fi

echo
echo "Refresh bundle: ${REPORT_DIR}"
echo "Read first: ${FIRST_READ}"
