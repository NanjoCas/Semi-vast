#!/usr/bin/env bash
# 全量实验 + 跑完自动关机（README_v3 12.5）：ratio 0.1 / 0.25 / 0.5 × A / B / F / O × seed 42–46，config_fc（detector 编译）。
#   1. 等 WAIT_PID（正在跑的 run_main_fc.sh）结束，再执行一次 run_main_fc.sh：已完成的 detector 跳过，中途失败的从断点继续；
#   2. ratio 0.25、0.5 各调用一次 run_all.py（从 SSL 划分、extractor 开始），一个失败不影响下一个；
#   3. 汇总全部 ratio（results_fc/summary.md）+ ratio 0.1 与 runs_f 的配对比较（results_fc/compare_runs.md）；
#   4. 无论成功失败都执行 shutdown（项目负责人 2026-11-06 决定）。日志与结果在数据盘上，关机后保留。
#
# 后台运行（与终端 / 会话脱离）：
#   cd /root/autodl-tmp/Semi-vast/v3 && WAIT_PID=<pid> setsid nohup bash run_full_fc.sh > logs/full_fc.log 2>&1 < /dev/null &
# 每个阶段的开始 / 结束与退出码记在 logs/full_fc_status.txt。不想关机：NO_SHUTDOWN=1。
set -uo pipefail   # 不用 -e：某个阶段失败也要继续后面的阶段并关机
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
CONFIG=configs/config_fc.yaml
SEEDS=42,43,44,45,46
STATUS=logs/full_fc_status.txt
FAILED=0

log() { echo "[full_fc] $(date '+%F %T') $*" | tee -a "$STATUS"; }
phase() {  # phase <名称> <命令...>
  local name=$1; shift
  log "开始 $name"
  "$@"
  local rc=$?
  if [[ $rc -eq 0 ]]; then log "完成 $name"; else log "失败 $name（退出码 $rc）"; FAILED=1; fi
}

log "启动（config $CONFIG，NO_SHUTDOWN=${NO_SHUTDOWN:-0}）"
if [[ -n "${WAIT_PID:-}" ]]; then
  log "等待 ratio 0.1 的进程 $WAIT_PID 结束"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

phase "ratio 0.1（补跑 / 确认）" bash run_main_fc.sh
for ratio in 0.25 0.5; do
  phase "ratio $ratio" python run_all.py --config "$CONFIG" --ratios "$ratio" --seeds "$SEEDS" --methods A,B,F,O
done
phase "汇总全部 ratio" python run_all.py --config "$CONFIG" --aggregate_only
phase "ratio 0.1 配对比较（runs_f vs runs_fc）" python evaluation/compare_runs.py --ratio 0.1

log "全部结束：$([[ $FAILED -eq 0 ]] && echo '没有失败的阶段' || echo '有失败的阶段，见上面的记录和 logs/full_fc.log')"
if [[ "${NO_SHUTDOWN:-0}" != "1" ]]; then
  log "关机"
  sync
  shutdown
fi
