#!/bin/bash
# 查看进度快照：bash logs/progress.sh [日志文件]，默认看 logs/ 下最新的日志；配合 watch -n 10 使用。
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=${1:-$(ls -t logs/*.log 2>/dev/null | head -1)}
[[ -z "$LOG" ]] && { echo "logs/ 下没有日志"; exit 1; }
echo "日志: $LOG    当前时间: $(date +%T)"
echo "---- 步骤 ----"
grep -a -E "^\[(STEP|DONE|FAIL)\]|SUCCESS" "$LOG" | tail -20
echo "---- 当前 detector 的验证记录 ----"
tr '\r' '\n' < "$LOG" | awk '/^\[STEP\]/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' \
  | grep -a -E "budget:|\[step [0-9]+/[0-9]+\]|New best|\[test\]" | sed -E 's/^[0-9:]+ \[INFO\] train_detector \| //'
echo "---- 实时进度 ----"
tr '\r' '\n' < "$LOG" | grep -a -E "^train:|Epoch [0-9]+/[0-9]+:|val@|test:" | tail -1
echo "---- 自动检查的警告 ----"
grep -a "\[SANITY\] ⚠️" "$LOG" | tail -10
echo "---- GPU ----"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
pgrep -f "train_detector.py|train_extractor.py|generate_pseudolabels.py" >/dev/null && echo "状态: 运行中" || echo "状态: 没有训练进程"
