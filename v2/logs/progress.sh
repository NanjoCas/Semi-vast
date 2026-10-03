#!/bin/bash
# 查看 detector 训练进度：bash logs/progress.sh [日志文件]，默认看最新的日志
cd /root/autodl-tmp/Semi-vast/v2
LOG=${1:-$(ls -t logs/*.log | head -1)}
echo "日志: $LOG    当前时间: $(date +%T)"
echo "---- 步骤 ----"
grep -a -E "^\[(STEP|DONE|FAIL)\]|SUCCESS" "$LOG"
echo "---- 当前组的 epoch ----"
tr '\r' '\n' < "$LOG" | awk '/^\[STEP\]/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' \
  | grep -a -E "epoch_mode=|Epoch [0-9]+/[0-9]+ |New best|\[test\]" | sed -E 's/ -> .*//; s/^[0-9:]+ \[INFO\] train_detector \| //'
echo "---- 实时进度 ----"
tr '\r' '\n' < "$LOG" | grep -a -E "train epoch|val/epoch|test:" | tail -1
echo "---- GPU ----"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
pgrep -f train_detector.py >/dev/null && echo "状态: 运行中" || echo "状态: 未在运行"
