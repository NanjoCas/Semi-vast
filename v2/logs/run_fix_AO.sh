#!/bin/bash
# Validate the detector budget / dynamic padding fix on r0.10_s42 (A and O only).
cd /root/autodl-tmp/Semi-vast/v2
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
for m in A O; do
  echo "[STEP] detector $m"; t0=$(date +%s)
  python training/train_detector.py --run_dir runs/r0.10_s42_fix --method $m --seed 42 --device cuda || { echo "[FAIL] detector $m"; exit 1; }
  echo "[DONE] detector $m ($(( ($(date +%s)-t0)/60 )) min)"
done
echo "[SUCCESS] A/O validation finished"
