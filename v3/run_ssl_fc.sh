#!/usr/bin/env bash
# README_v3 第 14 节：标准半监督基线 FIX（FixMatch）、FREE（FreeMatch）× seed 42–46，ratio 0.1。
# 沿用 runs_fc 的划分，config_fc.yaml 不变，所以已有的 A / F 直接参与配对比较（检验 U1、U2）。
# 需要先运行 tools/backtranslate.py，并把选定的回译方式填进 common/paths.py 的 SSL_SETTINGS['bt_variant']。
# 约 2 小时（每个 detector 约 11 分钟）。最后汇总到 results_fc/summary.md（第 2c 节）。
#
# 后台运行并记录日志（换卡或开机后先跑 python tools/gpu_memtest.py）：
#   cd /root/autodl-tmp/Semi-vast/v3 && setsid nohup bash run_ssl_fc.sh > logs/ssl_fc.log 2>&1 < /dev/null &
#   bash logs/watch.sh logs/ssl_fc.log
# 中断后重新执行同一条命令即可继续（已完成且设置相同的 detector 会被跳过）。只跑一部分：METHODS=FIX bash run_ssl_fc.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
CONFIG=${CONFIG:-configs/config_fc.yaml}
METHODS=${METHODS:-FIX,FREE}
echo "[ssl_fc] $(date '+%F %T') config $CONFIG，方法 $METHODS"
python run_all.py --config "$CONFIG" --ratios 0.1 --seeds 42,43,44,45,46 --methods "$METHODS"
echo "[ssl_fc] $(date '+%F %T') 全部结束"
