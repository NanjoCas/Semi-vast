"""
compute_nli_probs.py (v3, 方向一)
=================================
用 LogicScore 的同一个 NLI 模型（models.nli_model），计算句对的三类概率，
顺序与伪标签一致：[P(蕴含), P(矛盾), P(中立)] = [SUPPORTS, REFUTES, NOT_ENOUGH_INFO]。
premise = 证据（前 5 条拼接，与 models/extractor.py 生成 LogicScore 时相同），hypothesis = claim。

两种用法：
  --run_dir runs_f/r0.10_s42   无标签池 → <run>/pseudo/nli_probs.jsonl（方法 F 的融合 teacher，README_v3 10.6）。
                               会核对 P(蕴含) − P(矛盾) 与伪标签池中保存的 LogicScore 是否一致，
                               不一致说明输入构造与生成伪标签池时不同，直接报错。
  --split test                 processed/labeled/test.jsonl → processed/nli/test.jsonl（A⊕NLI 对照）。

输出每行 {"id": ..., "probs": [p_sup, p_ref, p_nei]}。

Usage:
    python training/compute_nli_probs.py --config configs/config_f.yaml --run_dir runs_f/r0.10_s42
    python training/compute_nli_probs.py --config configs/config_f.yaml --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import load_jsonl, save_jsonl  # noqa: E402
from common.paths import RunPaths, cfg_path, labeled_split_path, load_config, nli_split_path  # noqa: E402
from models.logic_scorer import LogicScorer  # noqa: E402

MAX_EVIDENCES = 5            # 与 models/extractor.py 生成 LogicScore 时相同（ClaimEvidenceDataset.max_evidences）
MAX_LOGIC_SCORE_DIFF = 0.02  # 与伪标签池中 LogicScore 的最大允许差异（只有数值误差时约 1e-6）


def premise(rec: dict) -> str:
    evidences = rec.get("evidence", []) or []
    if isinstance(evidences, str):
        evidences = [evidences]
    return " ".join(evidences[:MAX_EVIDENCES])


def main() -> None:
    parser = argparse.ArgumentParser(description="NLI class probabilities for a run's unlabeled pool or a labeled split.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_dir", type=str, help="Score the run's unlabeled pool")
    group.add_argument("--split", type=str, choices=["train", "dev", "test"], help="Score a labeled split")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_dir:
        paths = RunPaths(args.run_dir)
        records, out_path = load_jsonl(paths.unlabeled_pool), paths.nli_probs
    else:
        records, out_path = load_jsonl(labeled_split_path(cfg, args.split)), nli_split_path(cfg, args.split)
    missing = sum(1 for r in records if not premise(r).strip())
    if missing:
        print(f"[compute_nli_probs] WARNING: {missing} records have no evidence")

    scorer = LogicScorer(
        model_name=cfg["models"]["nli_model"],
        device=args.device,
        cache_dir=cfg_path(cfg, "model_cache_dir"),
        local_files_only=bool(cfg.get("use_local_models", False)),
    )
    probs = scorer.probs_batch([(r["claim"], premise(r)) for r in records], batch_size=args.batch_size)
    rows = [{"id": str(r["id"]), "probs": [round(float(x), 6) for x in p]} for r, p in zip(records, probs)]

    if args.run_dir:
        stored = {str(r["id"]): float(r["logic_score"]) for r in load_jsonl(paths.pseudo_pool)}
        ids = [row["id"] for row in rows]
        if set(ids) != set(stored):
            raise SystemExit(f"[compute_nli_probs] 无标签池与伪标签池的 id 不一致：{len(set(ids) ^ set(stored))} 条不同")
        diff = np.abs(np.array([row["probs"][0] - row["probs"][1] - stored[row["id"]] for row in rows]))
        print(f"[compute_nli_probs] P(蕴含) − P(矛盾) 与伪标签池 LogicScore 的差异：max {diff.max():.2e}，mean {diff.mean():.2e}")
        if diff.max() > MAX_LOGIC_SCORE_DIFF:
            raise SystemExit(f"[compute_nli_probs] 差异超过 {MAX_LOGIC_SCORE_DIFF}：NLI 的输入构造与生成伪标签池时不同")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, out_path)
    mean = np.mean([row["probs"] for row in rows], axis=0)
    print(f"[compute_nli_probs] {len(rows)} pairs -> {out_path}；平均概率 SUP/REF/NEI = {mean.round(3).tolist()}")


if __name__ == "__main__":
    main()
