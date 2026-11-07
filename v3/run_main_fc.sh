#!/usr/bin/env bash
# 方向一 + torch.compile（README_v3 第 12 节）：A / B / F / O × seed 42–46，ratio 0.1，detector 用编译后的训练。
# 划分与伪标签池原样沿用 runs_f（extractor 和 [pool] 设置不变），所以与 runs_f 的唯一差别是 detector 是否编译，
# 两边逐 seed 的结果可以直接配对比较。约 3 小时（主机空闲时）。
#
# 后台运行并记录日志：
#   cd /root/autodl-tmp/Semi-vast/v3 && nohup bash run_main_fc.sh > logs/main_fc.log 2>&1 &
#   bash logs/watch.sh logs/main_fc.log
# 中断后重新执行同一条命令即可继续（已复制的文件不会覆盖，已完成且指纹相同的 detector 会被跳过）。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
CONFIG=${CONFIG:-configs/config_fc.yaml}
SOURCE_CONFIG=configs/config_f.yaml
SEEDS="42 43 44 45 46"
echo "[main_fc] config: $CONFIG（伪标签池来自 $SOURCE_CONFIG）"

python - "$SOURCE_CONFIG" "$CONFIG" <<'EOF'
import sys
from common.fingerprint import diff_settings, pool_settings
from common.paths import load_config
diff = diff_settings(pool_settings(load_config(sys.argv[1])), pool_settings(load_config(sys.argv[2])))
if diff:
    sys.exit("[main_fc] [pool] 设置与伪标签池来源不同，不能沿用：\n  " + "\n  ".join(diff))
print("[main_fc] [pool] 设置一致，沿用伪标签池")
EOF

SRC=$(python -c "import sys; from common.paths import load_config, cfg_path; print(cfg_path(load_config(sys.argv[1]), 'runs_dir'))" "$SOURCE_CONFIG")
DST=$(python -c "import sys; from common.paths import load_config, cfg_path; print(cfg_path(load_config(sys.argv[1]), 'runs_dir'))" "$CONFIG")
for s in $SEEDS; do
  src="$SRC/r0.10_s$s"; dst="$DST/r0.10_s$s"
  [[ -f "$src/pseudo/pseudo_pool.jsonl" && -f "$src/pseudo/nli_probs.jsonl" ]] || { echo "[main_fc] 缺少 $src 的伪标签池"; exit 1; }
  mkdir -p "$dst/outputs"
  for item in data pseudo imported_from.json; do
    if [[ -e "$src/$item" && ! -e "$dst/$item" ]]; then cp -r "$src/$item" "$dst/"; fi
  done
  if [[ -d "$src/outputs/extractor" && ! -e "$dst/outputs/extractor" ]]; then cp -r "$src/outputs/extractor" "$dst/outputs/"; fi
  echo "[main_fc] r0.10_s$s：沿用 $src 的划分与伪标签池"
done

python run_all.py --config "$CONFIG" --ratios 0.1 --seeds "${SEEDS// /,}" --methods A,B,F,O
