"""
run_all.py — v2 一键运行脚本
============================

流程（对 config.experiment 中每个 label_ratio × seed）：
  0. build_labeled          修复泄漏后重建 processed/labeled/{train,dev,test}.jsonl（全局一次）
  1. make_ssl_split         切出有标签部分 + 无标签池（含 evidence）
  2. train_extractor        只用有标签部分训练 extractor
  3. generate_pseudolabels  对无标签句对打伪标签，LogicScore 使用真实证据
  4. train_rl_selector      PPO 选择 → 方法 C
  5. build_baseline_sets    方法 B / W / R / O 的伪标签集合
  6. train_detector × 方法  A / B / W / R / C / O
  7. pseudo_label_quality   用隐藏的金标签评估伪标签与选择质量
  8. 清理该 run 的 extractor / RL 权重（cleanup_checkpoints）
最后：aggregate_results 汇总到 results/

断点续跑：每一步的产物已存在就跳过；加 --force 重新跑全部步骤。

Usage:
    python run_all.py                                   # 按 config 跑完整实验矩阵
    python run_all.py --ratios 0.1 --seeds 42           # 先跑一个 run 看效果
    python run_all.py --methods A,C,R --device cuda
    python run_all.py --aggregate_only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from common.paths import RunPaths, labeled_split_path, load_config, run_dir  # noqa: E402

PYTHON = sys.executable
ENV_EXTRA = {
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
ALL_METHODS = ["A", "B", "W", "R", "C", "O"]


def _display(path: Path) -> str:
    """Path relative to v2/ when possible (paths in config may point elsewhere, e.g. another drive)."""
    try:
        return str(path.relative_to(V2_ROOT))
    except ValueError:
        return str(path)


def run_step(name: str, cmd: list[str], done_marker: Path | None, force: bool) -> None:
    if done_marker is not None and done_marker.exists() and not force:
        print(f"[SKIP] {name} (found {_display(done_marker)})")
        return
    print("\n" + "=" * 78)
    print(f"[STEP] {name}")
    print(f"[CMD ] {' '.join(cmd)}")
    print("=" * 78, flush=True)
    env = os.environ.copy()
    env.update(ENV_EXTRA)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(V2_ROOT), env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[FAIL] {name} exited with code {result.returncode} after {elapsed:.0f}s")
        print("       修复后重新运行 run_all.py 即可从这一步继续（已完成的步骤会被跳过）。")
        sys.exit(result.returncode)
    print(f"[DONE] {name} ({elapsed / 60:.1f} min)", flush=True)


def parse_list(value: str | None, cast) -> list | None:
    if value is None:
        return None
    return [cast(v) for v in value.replace(" ", "").split(",") if v]


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 one-click pipeline.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--ratios", type=str, default=None, help="Comma-separated, overrides experiment.label_ratios")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated, overrides experiment.seeds")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated subset of A,B,W,R,C,O")
    parser.add_argument("--device", type=str, default=None, help="Overrides experiment.device")
    parser.add_argument("--force", action="store_true", help="Re-run steps even if their outputs exist")
    parser.add_argument("--rebuild_data", action="store_true", help="Re-run build_labeled even if processed data exists")
    parser.add_argument("--keep_checkpoints", action="store_true", help="Do not delete .pt files after each run")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()

    config_arg = args.config
    cfg = load_config(config_arg)
    exp = cfg.get("experiment", {})
    ratios = parse_list(args.ratios, float) or [float(r) for r in exp.get("label_ratios", [0.1])]
    seeds = parse_list(args.seeds, int) or [int(s) for s in exp.get("seeds", [42])]
    methods = parse_list(args.methods, str) or list(exp.get("methods", ALL_METHODS))
    methods = [m.upper() for m in methods]
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    device = args.device or exp.get("device", "cuda")
    cleanup = bool(exp.get("cleanup_checkpoints", True)) and not args.keep_checkpoints

    if args.aggregate_only:
        run_step("Aggregate results", [PYTHON, "evaluation/aggregate_results.py", "--config", config_arg], None, True)
        return

    print(f"[INFO] v2 root   : {V2_ROOT}")
    print(f"[INFO] ratios    : {ratios}")
    print(f"[INFO] seeds     : {seeds}")
    print(f"[INFO] methods   : {methods}")
    print(f"[INFO] device    : {device}")
    print(f"[INFO] runs      : {len(ratios) * len(seeds)}  detectors per run: {len(methods)}")

    # ---- 0. leakage-free labeled data (global) ----
    run_step(
        "Build labeled data (leakage fixed)",
        [PYTHON, "data_prep/build_labeled.py", "--config", config_arg],
        None if args.rebuild_data else labeled_split_path(cfg, "test"),
        args.force or args.rebuild_data,
    )

    needs_pseudo = any(m != "A" for m in methods)
    needs_rl = any(m in ("C", "R") for m in methods)
    baseline_methods = [m for m in methods if m in ("B", "W", "R", "O")]

    for ratio in ratios:
        for seed in seeds:
            rd = run_dir(cfg, ratio, seed)
            paths = RunPaths(rd)
            tag = rd.name
            common = ["--config", config_arg, "--run_dir", str(rd)]
            print(f"\n{'#' * 78}\n# RUN {tag}\n{'#' * 78}")

            run_step(f"[{tag}] SSL split",
                     [PYTHON, "data_prep/make_ssl_split.py", "--config", config_arg,
                      "--ratio", str(ratio), "--seed", str(seed), "--run_dir", str(rd)],
                     paths.unlabeled_gold, args.force)

            if needs_pseudo:
                pseudo_done = paths.pseudo_pool.exists() and not args.force
                rl_done = paths.rl_selected.exists() and not args.force
                if not (pseudo_done and (rl_done or not needs_rl)):
                    run_step(f"[{tag}] Train extractor",
                             [PYTHON, "training/train_extractor.py", *common, "--seed", str(seed), "--device", device],
                             paths.extractor_ckpt, args.force)
                run_step(f"[{tag}] Generate pseudo labels",
                         [PYTHON, "training/generate_pseudolabels.py", *common, "--device", device],
                         paths.pseudo_pool, args.force)
                if needs_rl:
                    run_step(f"[{tag}] Train RL selector",
                             [PYTHON, "training/train_rl_selector.py", *common, "--seed", str(seed), "--device", device],
                             paths.rl_selected, args.force)
                if baseline_methods:
                    markers = [paths.pseudo_set(m) for m in baseline_methods]
                    all_built = all(p.exists() for p in markers)
                    run_step(f"[{tag}] Build baseline pseudo sets ({','.join(baseline_methods)})",
                             [PYTHON, "training/build_baseline_sets.py", *common, "--seed", str(seed),
                              "--methods", ",".join(baseline_methods)],
                             markers[0] if all_built else None, args.force)

            for method in methods:
                cmd = [PYTHON, "training/train_detector.py", *common, "--method", method,
                       "--seed", str(seed), "--device", device]
                if not cleanup:
                    cmd.append("--keep_checkpoint")
                run_step(f"[{tag}] Detector method {method}", cmd,
                         paths.detector_outputs(method) / "test_results.json", args.force)

            if needs_pseudo:
                run_step(f"[{tag}] Pseudo-label quality",
                         [PYTHON, "evaluation/pseudo_label_quality.py", "--run_dir", str(rd)],
                         None, True)

            if cleanup and paths.checkpoints.exists():
                shutil.rmtree(paths.checkpoints, ignore_errors=True)
                print(f"[CLEAN] removed {_display(paths.checkpoints)}")

    run_step("Aggregate results", [PYTHON, "evaluation/aggregate_results.py", "--config", config_arg], None, True)
    print("\n[SUCCESS] v2 pipeline finished. See results/summary.md and results/significance.csv")


if __name__ == "__main__":
    main()
