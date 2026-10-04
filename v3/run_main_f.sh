#!/usr/bin/env bash
# 方向一主实验（README_v3 10.8）：A / B / F / O × seed 42–46，ratio 0.1；A⊕NLI 对照在汇总时自动生成（不训练）。约 6 小时（主机空闲时）。
# 试跑（run_pilot_f.sh）的 G1 未通过（F − A = +0.046 / +0.013，阈值 0.02），2026-10-04 项目负责人决定仍进入主实验（README_v3 10.7.1、10.8），
# 所以这里不检查 pilot_report_f.json。试跑中已完成的 seed 42 的 A / F / O 会被跳过（配置指纹相同）。
#
# 后台运行并记录日志：
#   cd /root/autodl-tmp/Semi-vast/v3 && nohup bash run_main_f.sh > logs/main_f.log 2>&1 &
#   bash logs/watch.sh logs/main_f.log
# 中断后重新执行同一条命令即可继续（已完成且指纹相同的 detector 会被跳过）。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
CONFIG=${CONFIG:-configs/config_f.yaml}
echo "[main_f] config: $CONFIG"

python tools/import_from_v2.py --config "$CONFIG" --seeds 42,43,44
python run_all.py --config "$CONFIG" --ratios 0.1 --seeds 42,43,44,45,46 --methods A,B,F,O
