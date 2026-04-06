#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CUDA_TAG="${CUDA_TAG:-cu128}"
HF_MODEL_ID="${HF_MODEL_ID:-microsoft/deberta-v3-base}"
HF_LOCAL_DIR="${HF_LOCAL_DIR:-$PROJECT_ROOT/.hf_models/deberta-v3-base}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python not found: $PYTHON_BIN"
  exit 1
fi

if [[ ! -f "requirements.txt" ]]; then
  echo "[ERROR] requirements.txt not found in $PROJECT_ROOT"
  exit 1
fi

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Venv: $VENV_DIR"

echo "[STEP] Create virtual environment"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[STEP] Upgrade pip/setuptools/wheel"
pip install --upgrade pip setuptools wheel

echo "[STEP] Install dependencies (CUDA: $CUDA_TAG)"
# requirements.txt contains torch==...+cu128, so we use PyTorch index + fallback extra index.
pip install -r requirements.txt \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  --extra-index-url "https://pypi.org/simple"

echo "[STEP] Verify core imports"
python - <<'PY'
import torch, transformers, sklearn, pandas, numpy
print('[OK] torch:', torch.__version__)
print('[OK] cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[OK] gpu:', torch.cuda.get_device_name(0))
print('[OK] transformers:', transformers.__version__)
print('[OK] sklearn:', sklearn.__version__)
print('[OK] pandas:', pandas.__version__)
print('[OK] numpy:', numpy.__version__)
PY

if [[ "$SKIP_MODEL_DOWNLOAD" != "1" ]]; then
  echo "[STEP] Download local HF model snapshot"
  mkdir -p "$(dirname "$HF_LOCAL_DIR")"
  python - <<PY
from huggingface_hub import snapshot_download
local_dir = r"${HF_LOCAL_DIR}"
repo_id = r"${HF_MODEL_ID}"
path = snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
print(f"[OK] model cached at: {path}")
PY
else
  echo "[SKIP] Model download (SKIP_MODEL_DOWNLOAD=1)"
fi

echo "[SUCCESS] Environment setup completed."
echo "[INFO] Activate with: source ${VENV_DIR}/bin/activate"
