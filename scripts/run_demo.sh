#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src python3 -m t1d_virtual_cohort.cli demo \
  --config configs/analysis.yaml \
  --output outputs/demo

