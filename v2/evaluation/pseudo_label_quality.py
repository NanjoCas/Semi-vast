"""
pseudo_label_quality.py
=======================
用无标签池被隐藏的金标签，直接评估伪标签与各选择策略的质量：

  1. 整个池的伪标签准确率（总体 / 分类别 / 混淆矩阵）
  2. 每个方法的伪标签集合（B / W / R / C）的规模、准确率、类别分布
  3. 各打分信号区分"伪标签对 / 错"的能力（AUROC）：
       confidence、-entropy、|LogicScore|、discourse_score、复合 weight
     |LogicScore| 的 AUROC ≈ 0.5 说明它对挑选正确伪标签没有帮助。

这一步只读取金标签做评估，不影响任何训练。

Usage:
    python evaluation/pseudo_label_quality.py --run_dir runs/r0.10_s42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))

from common.data_utils import ID2LABEL, LABEL2ID, label_to_id, load_jsonl  # noqa: E402
from common.paths import RunPaths  # noqa: E402


def accuracy_report(records: list[dict], gold: dict[str, int]) -> dict:
    pairs = [(gold[str(r["id"])], label_to_id(r["pseudo_label"])) for r in records if str(r["id"]) in gold]
    if not pairs:
        return {"size": len(records), "matched": 0}
    y_true = np.array([p[0] for p in pairs])
    y_pred = np.array([p[1] for p in pairs])
    per_class = {}
    for cid, name in ID2LABEL.items():
        mask = y_pred == cid
        per_class[name] = {
            "predicted": int(mask.sum()),
            "precision": float((y_true[mask] == cid).mean()) if mask.any() else None,
        }
    return {
        "size": len(records),
        "matched": len(pairs),
        "accuracy": float((y_true == y_pred).mean()),
        "per_pseudo_class": per_class,
        "confusion_matrix(rows=gold,cols=pseudo)": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def signal_auroc(pool: list[dict], gold: dict[str, int]) -> dict:
    rows = [r for r in pool if str(r["id"]) in gold]
    correct = np.array([int(label_to_id(r["pseudo_label"]) == gold[str(r["id"])]) for r in rows])
    if len(rows) == 0 or correct.min() == correct.max():
        return {"note": "AUROC undefined (all pseudo labels correct or all wrong)"}
    signals = {
        "confidence": [float(r.get("confidence", 0.0)) for r in rows],
        "neg_entropy": [-float(r.get("entropy", 0.0)) for r in rows],
        "abs_logic_score": [abs(float(r.get("logic_score", 0.0))) for r in rows],
        "discourse_score": [float(r.get("discourse_score", 0.0)) for r in rows],
        "composite_weight": [float(r.get("weight", 0.0)) for r in rows],
    }
    return {name: float(roc_auc_score(correct, np.array(vals))) for name, vals in signals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pseudo-label quality against hidden gold labels.")
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    paths = RunPaths(args.run_dir)
    gold = {str(g["id"]): LABEL2ID[g["label"]] for g in load_jsonl(paths.unlabeled_gold)}
    pool = load_jsonl(paths.pseudo_pool)

    report: dict = {
        "run": paths.root.name,
        "full_pool": accuracy_report(pool, gold),
        "signal_auroc_for_correctness": signal_auroc(pool, gold),
        "methods": {},
    }
    if paths.pseudo_filtered.exists():
        report["filtered_pool(weight>=threshold)"] = accuracy_report(load_jsonl(paths.pseudo_filtered), gold)
    for method in ("B", "W", "R", "C"):
        p = paths.pseudo_set(method)
        if p is not None and p.exists():
            report["methods"][method] = accuracy_report(load_jsonl(p), gold)

    out = paths.outputs / "pseudo_label_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"[pseudo_label_quality] {paths.root.name}")
    print(f"  full pool accuracy : {report['full_pool'].get('accuracy', float('nan')):.4f} (n={report['full_pool']['size']})")
    for m, r in report["methods"].items():
        print(f"  set {m:<2} accuracy    : {r.get('accuracy', float('nan')):.4f} (n={r['size']})")
    print(f"  signal AUROC       : {report['signal_auroc_for_correctness']}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
