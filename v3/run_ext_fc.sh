#!/usr/bin/env bash
# README_v3 第 13 节：只看 claim 的基线 AC、α 敏感性 F25 / F75 / F100、多 NLI 集成 E × seed 42–46，ratio 0.1。
# 沿用 runs_fc 的划分与伪标签池（不重训 extractor），config_fc.yaml 不变，所以已有的 A / B / F / O 直接参与配对比较。
# 先在验证集上检查 3 个 NLI 模型的零样本 macro-F1（低于 0.35 即停），再算另外两个模型对每个池的概率。
# 约 3.5 小时（主机空闲时）。最后汇总到 results_fc/summary.md（第 2b 节）。
#
# 后台运行并记录日志：
#   cd /root/autodl-tmp/Semi-vast/v3 && setsid nohup bash run_ext_fc.sh > logs/ext_fc.log 2>&1 < /dev/null &
#   bash logs/watch.sh logs/ext_fc.log
# 中断后重新执行同一条命令即可继续（已完成且指纹相同的 detector 会被跳过）。只跑一部分：METHODS=AC bash run_ext_fc.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
CONFIG=${CONFIG:-configs/config_fc.yaml}
METHODS=${METHODS:-AC,F25,F75,F100,E}
echo "[ext_fc] $(date '+%F %T') config $CONFIG，方法 $METHODS"
python run_all.py --config "$CONFIG" --ratios 0.1 --seeds 42,43,44,45,46 --methods "$METHODS"
echo "[ext_fc] $(date '+%F %T') 全部结束"
