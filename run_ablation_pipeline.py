"""
run_ablation_pipeline.py
=========================
One-click pipeline for the two remaining ablation baselines from
PROJECT_README.md Section 8:

    Baseline A: Supervised only (no pseudo-labels)
    Baseline B: Semi-supervised with confidence threshold (no RL, no LogicScore)

(Baseline C, the full model, is training/train_detector.py's default run and
is assumed to already exist at outputs/final_test_results.json /
checkpoints/detector/best_model.pt — produced by the normal Phase 1-4 pipeline.)

Both baselines reuse the already-trained extractor (Phase 1) and the raw
pseudo-label pool (Phase 2 output, processed/pseudo_labels/pseudo_labeled_pool.jsonl).
They only change how the Phase 4 detector's pseudo-labeled content channel is
built:
    - Baseline A: content channel disabled entirely (training.use_pseudo=false)
    - Baseline B: content channel built by a plain confidence threshold
      (training/build_confidence_baseline.py), bypassing LogicScore,
      DiscourseScore and the PPO Reinforced Selector, with uniform weight=1.0.

Each baseline trains into its own checkpoints/detector_<run_name>/ and
outputs/<run_name>/ directory (via train_detector.py --run_name), so nothing
overwrites the full-model (Baseline C) results already on disk.

Usage:
    python run_ablation_pipeline.py
    python run_ablation_pipeline.py --confidence_threshold 0.7 --device cuda
    python run_ablation_pipeline.py --skip_a --skip_b   # just regenerate the comparison plot
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
PYTHON = sys.executable

RUN_ENV_EXTRA = {
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ablation Baselines A and B, then compare against the full model (Baseline C).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Confidence threshold used to build Baseline B's pseudo-label set.",
    )
    parser.add_argument("--skip_a", action="store_true", help="Skip Baseline A training.")
    parser.add_argument("--skip_b", action="store_true", help="Skip Baseline B training.")
    parser.add_argument(
        "--skip_compare", action="store_true", help="Skip the final comparison plot/table."
    )
    return parser.parse_args()


def run_step(name: str, cmd: list[str]) -> None:
    print("\n" + "=" * 72)
    print(f"[STEP] {name}")
    print(f"[CMD ] {' '.join(cmd)}")
    print("=" * 72)
    t0 = time.time()
    import os

    env = os.environ.copy()
    env.update(RUN_ENV_EXTRA)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[FAIL] {name} exited with code {result.returncode} after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"[DONE] {name} ({elapsed:.1f}s)")


def build_baseline_a_config(base_config_path: Path) -> Path:
    """Copy config.yaml with training.use_pseudo forced to false."""
    import yaml

    with open(base_config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("training", {})["use_pseudo"] = False

    out_path = PROJECT_ROOT / "configs" / "ablation_baseline_A.yaml"
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=True)
    print(f"[INFO] Wrote Baseline A config (use_pseudo=false) to {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    base_config_path = PROJECT_ROOT / args.config

    # ---- Prerequisite checks ----
    required = [
        PROJECT_ROOT / "processed" / "labeled" / "train.jsonl",
        PROJECT_ROOT / "processed" / "labeled" / "dev.jsonl",
        PROJECT_ROOT / "processed" / "labeled" / "test.jsonl",
        PROJECT_ROOT / "processed" / "pseudo_labels" / "pseudo_labeled_pool.jsonl",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("[ERROR] Missing required input file(s):")
        for p in missing:
            print(f"  - {p}")
        print(
            "Run run_pipeline.py and training/generate_pseudolabels.py "
            "(Phase 1/2) before this ablation pipeline."
        )
        sys.exit(1)

    full_model_results = PROJECT_ROOT / "outputs" / "final_test_results.json"
    if not full_model_results.exists():
        print(
            f"[WARN] {full_model_results} not found — the comparison step will be "
            "missing Baseline C (full model). Run training/train_detector.py with "
            "the default (no --run_name) config first if you want the 3-way comparison."
        )

    # ---- Baseline A: supervised only ----
    if not args.skip_a:
        baseline_a_config = build_baseline_a_config(base_config_path)
        run_step(
            "Baseline A — supervised only (no pseudo-labels)",
            [
                PYTHON,
                "training/train_detector.py",
                "--config",
                str(baseline_a_config.relative_to(PROJECT_ROOT)),
                "--device",
                args.device,
                "--run_name",
                "baseline_A_supervised",
            ],
        )
    else:
        print("[SKIP] Baseline A")

    # ---- Baseline B: confidence threshold only (no RL, no LogicScore) ----
    if not args.skip_b:
        run_step(
            "Build Baseline B pseudo-label set (confidence threshold only)",
            [
                PYTHON,
                "training/build_confidence_baseline.py",
                "--threshold",
                str(args.confidence_threshold),
            ],
        )
        run_step(
            "Baseline B — semi-supervised, confidence threshold only",
            [
                PYTHON,
                "training/train_detector.py",
                "--config",
                args.config,
                "--device",
                args.device,
                "--run_name",
                "baseline_B_confidence",
                "--pseudo_path",
                "processed/pseudo_labels/baseline_B_confidence_filtered.jsonl",
            ],
        )
    else:
        print("[SKIP] Baseline B")

    # ---- Comparison ----
    if not args.skip_compare:
        run_step(
            "Comparing Baseline A / B / C (full model)",
            [PYTHON, "evaluation/compare_ablation_baselines.py"],
        )
    else:
        print("[SKIP] Comparison")

    print("\n[SUCCESS] Ablation pipeline finished.")
    print("  Baseline A results : outputs/baseline_A_supervised/final_test_results.json")
    print("  Baseline B results : outputs/baseline_B_confidence/final_test_results.json")
    print("  Baseline C results : outputs/final_test_results.json")
    print("  Comparison         : outputs/ablation/ablation_comparison.{json,png,pdf}")


if __name__ == "__main__":
    main()
