#!/usr/bin/env bash
# README_v3 13.5.1：TAPT 可行性测试（seed 42，只看 MLM loss 与验证集 macro-F1，不加载测试集）。
# 每组 (学习率, 遮盖比例) 训练一次 TAPT（10 个 epoch，学习率不衰减），在第 1 / 3 / 10 个 epoch 存 encoder；
# 每个快照按 A 的设置微调（train_detector.py --method AT --skip_test），结果在 runs_fc/r0.10_s42/outputs/tapt_probe/。
# 快照微调完即删除权重（每个约 1.7GB），只留 tapt_info.json。约 1.6 小时（主机空闲时）。
#
#   cd /root/autodl-tmp/Semi-vast/v3 && setsid nohup bash tools/tapt_probe.sh > logs/tapt_probe.log 2>&1 < /dev/null &
#
# 补跑（13.5.1）：GRID=3e-5:0.15 EPOCHS=100 SAVE_EPOCHS=10,20,30,50,100 bash tools/tapt_probe.sh（目录名加 _e100）。
# 中断后重新执行即可继续（已有 val_results.json 的快照跳过）。跑完用 tools/tapt_probe_report.py 汇总并按 13.5.1 的规则选设置。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONFIG=${CONFIG:-configs/config_fc.yaml}
RUN=runs_fc/r0.10_s42
SEED=42
EPOCHS=${EPOCHS:-10}
SAVE_EPOCHS=${SAVE_EPOCHS:-1,3,10}
GRID=${GRID:-"3e-6:0.15 1e-5:0.15 3e-5:0.15 1e-5:0.30"}

echo "[tapt_probe] $(date '+%F %T') grid $GRID，epochs $SAVE_EPOCHS"
for item in $GRID; do
  lr=${item%%:*}; mask=${item##*:}
  name="lr${lr}_m${mask}"
  [[ $EPOCHS == 10 ]] || name="${name}_e${EPOCHS}"   # 补跑更长的训练（13.5.1）
  ckpt="$RUN/checkpoints/tapt_probe/$name"
  out="$RUN/outputs/tapt_probe/$name"
  todo=0
  for e in ${SAVE_EPOCHS//,/ }; do
    [[ -f "$out/epoch$e/val_results.json" ]] || todo=1
  done
  if [[ $todo == 0 ]]; then
    echo "[SKIP] $name（全部快照已有验证结果）"; continue
  fi
  echo "[STEP] $(date '+%T') TAPT $name"
  python training/tapt.py --config "$CONFIG" --run_dir "$RUN" --seed $SEED --lr "$lr" --mask_prob "$mask" \
    --epochs $EPOCHS --save_epochs $SAVE_EPOCHS --out_dir "$ckpt"
  for e in ${SAVE_EPOCHS//,/ }; do
    mkdir -p "$out/epoch$e"
    cp "$ckpt/epoch$e/tapt_info.json" "$out/epoch$e/"
    echo "[STEP] $(date '+%T') 微调 $name epoch $e"
    python training/train_detector.py --config "$CONFIG" --run_dir "$RUN" --method AT --seed $SEED --device cuda \
      --encoder_dir "$ckpt/epoch$e" --skip_test --out_dir "$out/epoch$e"
    rm -f "$ckpt/epoch$e/model.safetensors"
  done
  echo "[DONE] $(date '+%T') $name"
done
python tools/tapt_probe_report.py --run_dir "$RUN"
echo "[tapt_probe] $(date '+%F %T') 全部结束"
