#!/usr/bin/env bash
# Helper script to create dataset folders and remind where to fetch data.
# Actual downloads are manual because the sources require license/ToS acceptance.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$ROOT_DIR/datasets"

mkdir -p "$DATA_DIR/climate_fever" "$DATA_DIR/pubhealth" "$DATA_DIR/climate_twitter"

cat <<'EOF'
[download_datasets]
Manual download required (see project README references):
  1) Climate-FEVER JSONL:   https://github.com/tdiggelm/climate-fever-dataset
     -> place files into datasets/climate_fever/
  2) PUBHEALTH TSV:         https://github.com/neemakot/Health-Fact-Checking
     -> place files into datasets/pubhealth/
  3) ClimateMiSt / Twitter: DOI 10.21227/cdaz-jh77 (tweets/news JSON/CSV)
     -> place files into datasets/climate_twitter/

After downloading, run: python run_pipeline.py
EOF
