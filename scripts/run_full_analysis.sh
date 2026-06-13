#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REAL_SUMMARY="${REAL_SUMMARY:-data/raw/real_summary.csv}"
VIRTUAL_SUMMARY="${VIRTUAL_SUMMARY:-data/derived/virtual_profile_summary.csv}"
REAL_TRACES="${REAL_TRACES:-data/raw/real_cgm}"
VIRTUAL_TRACES="${VIRTUAL_TRACES:-data/raw/virtual_traces}"

for required in "$REAL_SUMMARY" "$VIRTUAL_SUMMARY"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

PYTHONPATH=src python3 -m t1d_virtual_cohort.cli run \
  --real-summary "$REAL_SUMMARY" \
  --virtual-summary "$VIRTUAL_SUMMARY" \
  --real-traces "$REAL_TRACES" \
  --virtual-traces "$VIRTUAL_TRACES" \
  --config configs/analysis.yaml \
  --output outputs
