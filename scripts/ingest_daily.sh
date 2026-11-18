#!/usr/bin/env bash
# Run at most once per ~2 h (CelesTrak fair use). Each run appends a dated snapshot to orbital_element_history,
# which is what makes chronological forecasting targets and the SGP4 back-test possible.
# Example cron (daily at 06:10):  10 6 * * *  cd /path/to/orbitalx && ./scripts/ingest_daily.sh
set -euo pipefail
python -m backend.ingest_cli --groups active cosmos-2251-debris iridium-33-debris fengyun-1c-debris cosmos-1408-debris
