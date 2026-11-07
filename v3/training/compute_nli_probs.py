"""
compute_nli_probs.py (v3, 方向一；第 13 节的方法 E)
==================================================
计算句对的 NLI 三类概率，顺序与伪标签一致：[P(蕴含), P(矛盾), P(中立)] = [SUPPORTS, REFUTES, NOT_ENOUGH_INFO]。
各模型的标签顺序不同，一律按模型配置里的 id2label 取列（models/logic_scorer.py），不写死下标。
premise = 证据（前 5 条拼接，与 models/extractor.py 生成 LogicScore 时相同），hypothesis = claim。

默认用 LogicScore 的同一个 NLI 模型（models.nli_model）；--model 可以换成方法 E 的另外两个模型
（common/paths.py 的 ENSEMBLE_NLI_MODELS），输出到各自的文件（文件名带模型名）。

两种用法：
  --run_dir runs_f/r0.10_s42   无标签池 → <run>/pseudo/nli_probs.jsonl（方法 F 的融合 teacher，README_v3 10.6）。
                               对 LogicScore 的模型，会核对 P(蕴含) − P(矛盾) 与伪标签池中保存的 LogicScore 是否一致，
                               不一致说明输入构造与生成伪标签池时不同，直接报错。
  --split test                 processed/labeled/test.jsonl → processed/nli/test.jsonl（A⊕NLI 对照）。
                               有标签的划分会打印零样本 macro-F1（整体与分来源）；加 --min_macro_f1 时低于它就报错
                               （README_v3 13.3：标签映射错误时 macro-F1 会接近随机）。

输出每行 {"id": ..., "probs": [p_sup, p_ref, p_nei]}。

Usage:
    python training/compute_nli_probs.py --config configs/config_f.yaml --run_dir runs_f/r0.10_s42
    python training/compute_nli_probs.py --config configs/config_f.yaml --split test
    python training/compute_nli_probs.py --config configs/config_fc.yaml --split dev --model facebook/bart-large-mnli --min_macro_f1 0.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import label_to_id, load_jsonl, save_jsonl  # noqa: E402
from common.paths import RunPaths, cfg_path, labeled_split_path, load_config, nli_split_path  # noqa: E402
from models.logic_scorer import LogicScorer  # noqa: E402

MAX_EVIDENCES = 5            # 与 models/extractor.py 生成 LogicScore 时相同（ClaimEvidenceDataset.max_evidences）
MAX_LOGIC_SCORE_DIFF = 0.02  # 与伪标签池中 LogicScore 的最大允许差异（只有数值误差时约 1e-6）
MAX_MODEL_LENGTH = 4096      # tokenizer 没有设定 model_max_length 时（默认值是一个极大的数）不会截断，直接报错


def premise(rec: dict) -> str:
    evidences = rec.get("evidence", []) or []
    if isinstance(evidences, str):
        evidences = [evidences]
    return " ".join(evidences[:MAX_EVIDENCES])


def zero_shot_report(records: list[dict], rows: list[dict]) -> dict[str, float]:
    """Zero-shot macro-F1 of argmax(NLI probs) against the gold labels, overall and per source."""
    gold = np.array([label_to_id(r["label"]) for r in records])
    pred = np.array([int(np.argmax(row["probs"])) for row in rows])
    src = np.array([r.get("source", "unknown") for r in records])
    out = {"overall": float(f1_score(gold, pred, average="macro", labels=[0, 1, 2], zero_division=0))}
    for s in sorted(set(src)):
        m = src == s
        out[s] = float(f1_score(gold[m], pred[m], average="macro", labels=[0, 1, 2], zero_division=0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="NLI class probabilities for a run's unlabeled pool or a labeled split.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_dir", type=str, help="Score the run's unlabeled pool")
    group.add_argument("--split", type=str, choices=["train", "dev", "test"], help="Score a labeled split")
    parser.add_argument("--model", type=str, default=None, help="NLI model (default: models.nli_model)")
    parser.add_argument("--min_macro_f1", type=float, default=None,
                        help="Labeled splits only: exit with an error when the zero-shot macro-F1 is below this")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    main_model = cfg["models"]["nli_model"]
    model_name = args.model or main_model
    is_main = model_name == main_model
    if args.run_dir:
        paths = RunPaths(args.run_dir)
        records, out_path = load_jsonl(paths.unlabeled_pool), paths.pool_nli_probs(None if is_main else model_name)
    else:
        records, out_path = load_jsonl(labeled_split_path(cfg, args.split)), nli_split_path(cfg, args.split, model_name)
    missing = sum(1 for r in records if not premise(r).strip())
    if missing:
        print(f"[compute_nli_probs] WARNING: {missing} records have no evidence")

    scorer = LogicScorer(
        model_name=model_name,
        device=args.device,
        cache_dir=cfg_path(cfg, "model_cache_dir"),
        local_files_only=bool(cfg.get("use_local_models", False)),
    )
    max_len = scorer.tokenizer.model_max_length
    if max_len > MAX_MODEL_LENGTH:
        raise SystemExit(f"[compute_nli_probs] {model_name} 的 tokenizer 没有设定 model_max_length（{max_len}），输入不会被截断")
    print(f"[compute_nli_probs] model {model_name}: id2label {scorer.model.config.id2label}, "
          f"columns (entailment, contradiction, neutral) = ({scorer.idx_entailment}, {scorer.idx_contradiction}, "
          f"{scorer.idx_neutral}), max_length {max_len}")
    probs = scorer.probs_batch([(r["claim"], premise(r)) for r in records], batch_size=args.batch_size)
    rows = [{"id": str(r["id"]), "probs": [round(float(x), 6) for x in p]} for r, p in zip(records, probs)]

    if args.run_dir and is_main:
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

    if args.split and all(r.get("label") is not None for r in records):
        report = zero_shot_report(records, rows)
        print(f"[compute_nli_probs] zero-shot macro-F1 ({args.split}, {model_name}): "
              + ", ".join(f"{k} {v:.3f}" for k, v in report.items()))
        if args.min_macro_f1 is not None and report["overall"] < args.min_macro_f1:
            out_path.unlink()  # 不留下结果，run_all.py 下次会重新检查
            raise SystemExit(f"[compute_nli_probs] 零样本 macro-F1 {report['overall']:.3f} < {args.min_macro_f1}："
                             "先核对这个模型的标签映射（README_v3 13.3）")


if __name__ == "__main__":
    main()
