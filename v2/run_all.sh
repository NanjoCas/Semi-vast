#!/usr/bin/env bash
# v2 一键运行（bash 包装）：激活上级目录的 .venv（如存在）后调用 run_all.py。
# 所有参数原样传给 run_all.py，例如：
#   bash run_all.sh --ratios 0.1 --seeds 42
set -euo pipefail

V2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$V2_DIR"

if [[ -f ../.venv/bin/activate ]]; then
  source ../.venv/bin/activate
elif [[ -f ../.venv/Scripts/activate ]]; then
  source ../.venv/Scripts/activate
fi

python run_all.py "$@"
