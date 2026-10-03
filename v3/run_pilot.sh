#!/usr/bin/env bash
# 第一阶段：试跑（README_v3 第 5.2 节）。只用 seed 42：A、B 各两个训练 seed（42 与 1042），再加 O。
# 结束时打印 5 条通过标准（C1–C5）的检查结果。约 1.5 小时（主机空闲时）。
#
# 后台运行并记录日志：
#   cd /root/autodl-tmp/Semi-vast/v3 && nohup bash run_pilot.sh > logs/pilot.log 2>&1 &
#   bash logs/watch.sh
# 换配置（例如试跑未通过、调整后重试）：CONFIG=configs/config_b.yaml nohup bash run_pilot.sh > logs/pilot_b.log 2>&1 &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
CONFIG=${CONFIG:-configs/config.yaml}
echo "[pilot] config: $CONFIG"

python tools/import_from_v2.py --config "$CONFIG" --seeds 42
python run_all.py --config "$CONFIG" --ratios 0.1 --seeds 42 --methods A,B --extra_train_seeds 1042
python run_all.py --config "$CONFIG" --ratios 0.1 --seeds 42 --methods O
python evaluation/pilot_report.py --config "$CONFIG" --ratio 0.1 --seed 42 --repeat_seed 1042
