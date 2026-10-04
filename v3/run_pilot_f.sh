#!/usr/bin/env bash
# 方向一试跑（README_v3 10.7）：logic-aware teacher（方法 F，extractor + NLI 融合，α = 0.5）。
# 只用 seed 42：A、F 各两个训练 seed（42 与 1042），再加 O；结束时打印通过标准 G1、G2 和参考项。约 1.5 小时（主机空闲时）。
#
# 后台运行并记录日志：
#   cd /root/autodl-tmp/Semi-vast/v3 && nohup bash run_pilot_f.sh > logs/pilot_f.log 2>&1 &
#   bash logs/watch.sh logs/pilot_f.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
CONFIG=${CONFIG:-configs/config_f.yaml}
echo "[pilot_f] config: $CONFIG"

python tools/import_from_v2.py --config "$CONFIG" --seeds 42
python run_all.py --config "$CONFIG" --ratios 0.1 --seeds 42 --methods A,F --extra_train_seeds 1042
python run_all.py --config "$CONFIG" --ratios 0.1 --seeds 42 --methods O
python evaluation/pilot_report_f.py --config "$CONFIG" --ratio 0.1 --seed 42 --repeat_seed 1042
