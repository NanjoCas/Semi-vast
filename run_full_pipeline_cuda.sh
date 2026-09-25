#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/config.yaml}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SKIP_PREP="${SKIP_PREP:-1}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config not found: $CONFIG_PATH"
  exit 1
fi
#环境选取
source .venv/bin/activate

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found. CUDA environment is unavailable."
  exit 1
fi

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print('[ERROR] torch.cuda.is_available() = False')
    sys.exit(1)
print(f'[OK] CUDA available: {torch.cuda.get_device_name(0)}')
PY

step() {
  local name="$1"
  shift
  echo
  echo "================================================================"
  echo "[STEP] $name"
  echo "[CMD ] $*"
  echo "================================================================"
  local t0
  t0=$(date +%s)
  "$@"
  local t1
  t1=$(date +%s)
  echo "[DONE] $name (${t1}s since epoch, elapsed $((t1 - t0))s)"
}

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [[ "$SKIP_PREP" != "1" ]]; then
  step "Data Processing Pipeline" \
    python run_pipeline.py
else
  echo "[SKIP] Data Processing Pipeline (SKIP_PREP=1)"
  for f in processed/labeled/train.jsonl processed/labeled/dev.jsonl processed/labeled/test.jsonl; do
    if [[ ! -s "$f" ]]; then
      echo "[ERROR] Required labeled file is missing or empty: $f"
      echo "[HINT ] You skipped preprocessing, so you must provide non-empty labeled JSONL files first."
      exit 1
    fi
  done
fi

step "Train Extractor" \
  python training/train_extractor.py --config "$CONFIG_PATH" --device cuda --batch_size 16

step "Generate Pseudo Labels" \
  python training/generate_pseudolabels.py --config "$CONFIG_PATH" --device cuda

step "Train RL Selector" \
  python training/train_rl_selector.py --config "$CONFIG_PATH" --device cuda

step "Train Detector" \
  python training/train_detector.py --config "$CONFIG_PATH" --device cuda

echo
echo "[SUCCESS] Full pipeline finished on CUDA."
