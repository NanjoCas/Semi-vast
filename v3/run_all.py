"""
run_all.py — v3 一键运行脚本（README_v3 第 5 节）
================================================

流程（对每个 label_ratio × seed）：
  0. build_labeled          重建 processed/labeled（全局一次；已存在则跳过）
  1. make_ssl_split         切出有标签部分 + 无标签池
  2. train_extractor        只用有标签部分训练 extractor（伪标签池已存在时跳过）
  3. generate_pseudolabels  对无标签池打伪标签，LogicScore 使用真实证据
  4. train_rl_selector      只有 C / R 需要（v2 设计）
  5. build_baseline_sets    B / Q / K / O（以及 W / R）的伪标签集合；每次都重建（几秒钟，结果确定）
  6. train_detector         每个方法 × 每个训练 seed；结果已存在且配置指纹相同才跳过
  7. pseudo_label_quality   用隐藏的金标签评估伪标签与各集合
  8. sanity_check           自动检查 S1–S7，只警告不中断
  9. 清理该 run 的 checkpoints/
最后：aggregate_results 汇总到 results/

防止混用配置：runs 目录下的 protocol.json 记录配置指纹（common/fingerprint.py）。
当前配置的指纹与记录不同时拒绝运行：请换一个 runs 目录（paths.runs_dir），或加 --force_protocol。

Usage:
    python run_all.py --dry_run                                   # 只打印将要执行的步骤
    python run_all.py --ratios 0.1 --seeds 42 --methods A,B --extra_train_seeds 1042   # 试跑的一部分
    python run_all.py                                             # 按 config 跑完整矩阵
    python run_all.py --aggregate_only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.fingerprint import config_fingerprint  # noqa: E402
from common.paths import (  # noqa: E402
    ALL_METHODS,
    LEGACY_METHODS,
    MAIN_METHODS,
    RunPaths,
    cfg_path,
    detector_tag,
    labeled_split_path,
    load_config,
    run_dir,
)

PYTHON = sys.executable
ENV_EXTRA = {
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
DRY_RUN = False


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_step(name: str, cmd: list[str], done_marker: Path | None, force: bool, fatal: bool = True) -> None:
    if done_marker is not None and done_marker.exists() and not force:
        print(f"[SKIP] {name} (found {_display(done_marker)})")
        return
    if DRY_RUN:
        print(f"[PLAN] {name}\n       {' '.join(cmd)}")
        return
    print("\n" + "=" * 78)
    print(f"[STEP] {name}")
    print(f"[CMD ] {' '.join(cmd)}")
    print("=" * 78, flush=True)
    env = os.environ.copy()
    env.update(ENV_EXTRA)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[FAIL] {name} exited with code {result.returncode} after {elapsed:.0f}s")
        if fatal:
            print("       修复后重新运行同一条命令即可从这一步继续（已完成的步骤会被跳过）。")
            sys.exit(result.returncode)
        return
    print(f"[DONE] {name} ({elapsed / 60:.1f} min)", flush=True)


def parse_list(value: str | None, cast) -> list | None:
    if value is None:
        return None
    return [cast(v) for v in value.replace(" ", "").split(",") if v]


def check_protocol(cfg: dict, config_arg: str, force_protocol: bool) -> str:
    """Refuse to mix configurations inside one runs directory (README_v3 6.1)."""
    fp = config_fingerprint(cfg)
    runs_dir = cfg_path(cfg, "runs_dir")
    proto = runs_dir / "protocol.json"
    if proto.exists():
        with open(proto, encoding="utf-8") as fh:
            recorded = json.load(fh).get("config_fingerprint")
        if recorded != fp:
            msg = (f"{_display(runs_dir)} 已经用另一份配置（指纹 {recorded}）跑过，当前配置的指纹是 {fp}。\n"
                   f"不同配置的结果不能放在同一个目录：请修改 paths.runs_dir / paths.results_dir 换一个新目录，"
                   f"或确认要覆盖时加 --force_protocol（已有的 detector 结果会因指纹不同而重跑）。")
            if not force_protocol:
                raise SystemExit("[PROTOCOL] " + msg)
            print("[PROTOCOL] ⚠️  " + msg.splitlines()[0] + " —— 已按 --force_protocol 覆盖记录。")
        else:
            print(f"[PROTOCOL] 配置指纹 {fp} 与 {_display(proto)} 一致")
    if not DRY_RUN:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with open(proto, "w", encoding="utf-8") as fh:
            json.dump({"config_fingerprint": fp, "config_file": config_arg,
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg},
                      fh, indent=2, ensure_ascii=False)
    return fp


def detector_done(out_dir: Path, fp: str) -> tuple[bool, str]:
    res_path = out_dir / "test_results.json"
    if not res_path.exists():
        return False, ""
    with open(res_path, encoding="utf-8") as fh:
        recorded = json.load(fh).get("config_fingerprint")
    if recorded == fp:
        return True, f"found {_display(res_path)}"
    return False, f"已有结果的配置指纹 {recorded} ≠ 当前 {fp}，重跑"


def main() -> None:
    global DRY_RUN
    parser = argparse.ArgumentParser(description="v3 one-click pipeline.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--ratios", type=str, default=None, help="Comma-separated, overrides experiment.label_ratios")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated split seeds, overrides experiment.seeds")
    parser.add_argument("--methods", type=str, default=None, help=f"Comma-separated subset of {','.join(ALL_METHODS)}")
    parser.add_argument("--extra_train_seeds", type=str, default=None,
                        help="Also train every listed method with these training seeds (outputs <method>_t<seed>), "
                             "e.g. 1042 for the pilot's training-noise check")
    parser.add_argument("--device", type=str, default=None, help="Overrides experiment.device")
    parser.add_argument("--force", action="store_true", help="Re-run steps even if their outputs exist")
    parser.add_argument("--force_protocol", action="store_true", help="Accept a config that differs from runs/protocol.json")
    parser.add_argument("--rebuild_data", action="store_true", help="Re-run build_labeled even if processed data exists")
    parser.add_argument("--keep_checkpoints", action="store_true", help="Do not delete checkpoints/ after each run")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print the plan without running anything")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    config_arg = args.config
    cfg = load_config(config_arg)
    exp = cfg.get("experiment", {})
    ratios = parse_list(args.ratios, float) or [float(r) for r in exp.get("label_ratios", [0.1])]
    seeds = parse_list(args.seeds, int) or [int(s) for s in exp.get("seeds", [42])]
    methods = [m.upper() for m in (parse_list(args.methods, str) or list(exp.get("methods", MAIN_METHODS)))]
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown} (v3 methods: {ALL_METHODS}; L was replaced by Q)")
    extra_train_seeds = parse_list(args.extra_train_seeds, int) or []
    device = args.device or exp.get("device", "cuda")
    cleanup = bool(exp.get("cleanup_checkpoints", True)) and not args.keep_checkpoints

    if args.aggregate_only:
        run_step("Aggregate results", [PYTHON, "evaluation/aggregate_results.py", "--config", config_arg], None, True)
        return

    fp = check_protocol(cfg, config_arg, args.force_protocol)
    print(f"[INFO] v3 root      : {ROOT}")
    print(f"[INFO] config       : {config_arg} (fingerprint {fp})")
    print(f"[INFO] ratios       : {ratios}")
    print(f"[INFO] seeds        : {seeds}")
    print(f"[INFO] methods      : {methods}")
    print(f"[INFO] train seeds  : split seed" + (f" + {extra_train_seeds}" if extra_train_seeds else ""))
    print(f"[INFO] device       : {device}")
    legacy = [m for m in methods if m in LEGACY_METHODS]
    if legacy:
        print(f"[WARN] {legacy} 仍是 v2 的设计（复合权重用 |LogicScore|、RL 奖励近似常数），尚未按 v3 修订，"
              "结果不能用于结论（README_v3 第 8 节）。")

    # ---- 0. labeled data (global) ----
    run_step(
        "Build labeled data",
        [PYTHON, "data_prep/build_labeled.py", "--config", config_arg],
        None if args.rebuild_data else labeled_split_path(cfg, "test"),
        args.force or args.rebuild_data,
    )

    needs_pseudo = any(m != "A" for m in methods)
    needs_rl = any(m in ("C", "R") for m in methods)
    baseline_methods = [m for m in methods if m in ("B", "Q", "K", "W", "R", "O")]

    for ratio in ratios:
        for seed in seeds:
            rd = run_dir(cfg, ratio, seed)
            paths = RunPaths(rd)
            tag_run = rd.name
            common = ["--config", config_arg, "--run_dir", str(rd)]
            print(f"\n{'#' * 78}\n# RUN {tag_run}" + ("  (imported from v2)" if paths.import_marker.exists() else "") + f"\n{'#' * 78}")

            run_step(f"[{tag_run}] SSL split",
                     [PYTHON, "data_prep/make_ssl_split.py", "--config", config_arg,
                      "--ratio", str(ratio), "--seed", str(seed), "--run_dir", str(rd)],
                     paths.unlabeled_gold, args.force)

            if needs_pseudo:
                pool_done = paths.pseudo_pool.exists() and not args.force
                rl_done = paths.rl_selected.exists() and not args.force
                if needs_rl and not rl_done and pool_done and not paths.extractor_ckpt.exists():
                    raise SystemExit(
                        f"[{tag_run}] C/R 需要生成伪标签池的同一个 extractor，但它的权重已不存在（跑完已清理，或池是从 v2 导入的）。"
                        "重新训练 extractor 会和现有的伪标签池不一致：请用新的 runs 目录，或对这个 run 加 --force 从头重跑。")
                if not pool_done or (needs_rl and not rl_done):
                    run_step(f"[{tag_run}] Train extractor",
                             [PYTHON, "training/train_extractor.py", *common, "--seed", str(seed), "--device", device],
                             paths.extractor_ckpt, args.force)
                run_step(f"[{tag_run}] Generate pseudo labels",
                         [PYTHON, "training/generate_pseudolabels.py", *common, "--device", device],
                         paths.pseudo_pool, args.force)
                if needs_rl:
                    run_step(f"[{tag_run}] Train RL selector (v2 design)",
                             [PYTHON, "training/train_rl_selector.py", *common, "--seed", str(seed), "--device", device],
                             paths.rl_selected, args.force)
                if baseline_methods:
                    run_step(f"[{tag_run}] Build pseudo sets ({','.join(baseline_methods)})",
                             [PYTHON, "training/build_baseline_sets.py", *common, "--seed", str(seed),
                              "--methods", ",".join(baseline_methods)],
                             None, True)

            for method in methods:
                for train_seed in [seed, *[s for s in extra_train_seeds if s != seed]]:
                    tag = detector_tag(method, seed, train_seed)
                    out_dir = paths.detector_outputs(tag)
                    done, why = detector_done(out_dir, fp)
                    if done and not args.force:
                        print(f"[SKIP] [{tag_run}] Detector {tag} ({why})")
                        continue
                    if why:
                        print(f"[INFO] [{tag_run}] Detector {tag}: {why}")
                    cmd = [PYTHON, "training/train_detector.py", *common, "--method", method,
                           "--seed", str(seed), "--train_seed", str(train_seed), "--device", device]
                    if not cleanup:
                        cmd.append("--keep_checkpoint")
                    run_step(f"[{tag_run}] Detector {tag}", cmd, None, True)

            if needs_pseudo:
                run_step(f"[{tag_run}] Pseudo-label quality",
                         [PYTHON, "evaluation/pseudo_label_quality.py", "--run_dir", str(rd)],
                         None, True, fatal=False)
            run_step(f"[{tag_run}] Sanity check",
                     [PYTHON, "evaluation/sanity_check.py", "--config", config_arg, "--run_dir", str(rd)],
                     None, True, fatal=False)

            if cleanup and paths.checkpoints.exists() and not DRY_RUN:
                shutil.rmtree(paths.checkpoints, ignore_errors=True)
                print(f"[CLEAN] removed {_display(paths.checkpoints)}")

    run_step("Aggregate results", [PYTHON, "evaluation/aggregate_results.py", "--config", config_arg], None, True)
    if not DRY_RUN:
        print(f"\n[SUCCESS] v3 pipeline finished. See {_display(cfg_path(cfg, 'results_dir'))}/summary.md")


if __name__ == "__main__":
    main()
