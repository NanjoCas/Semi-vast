#!/usr/bin/env bash
# 第二阶段：主实验（README_v3 第 5.3 节）。A / B / Q / K / O × seed 42–46，ratio 0.1。约 8–9 小时。
# 必须先通过试跑（run_pilot.sh 最后打印"全部通过"）；试跑中已完成的 seed 42 的 A / B / O 会被跳过。
#
# 后台运行并记录日志：
#   cd /root/autodl-tmp/Semi-vast/v3 && nohup bash run_main.sh > logs/main.log 2>&1 &
#   bash logs/watch.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
CONFIG=${CONFIG:-configs/config.yaml}
echo "[main] config: $CONFIG"

PILOT_JSON=$(python -c "import sys; sys.path.insert(0, '.'); from common.paths import load_config, run_dir; print(run_dir(load_config('$CONFIG'), 0.1, 42) / 'outputs' / 'pilot_report.json')")
if [[ ! -f "$PILOT_JSON" ]] || ! grep -q '"passed": true' "$PILOT_JSON"; then
  echo "[main] 试跑还没有通过（$PILOT_JSON），先运行 run_pilot.sh。" >&2
  exit 1
fi
python tools/import_from_v2.py --config "$CONFIG" --seeds 42,43,44
python run_all.py --config "$CONFIG" --ratios 0.1 --seeds 42,43,44,45,46 --methods A,B,Q,K,O
