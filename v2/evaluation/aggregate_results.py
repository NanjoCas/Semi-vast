"""
aggregate_results.py
====================
汇总所有 run 的测试结果，输出到 results/：

  summary.csv / summary.md     每个 (label_ratio, method) 的 mean ± std（跨 seed）
  significance.csv             配对 bootstrap：同一 run 内两个方法在同一测试集上的 ΔmacroF1
                               及 95% 置信区间（每个 seed 单独计算）
  pseudo_quality.csv           每个 run 的伪标签准确率与信号 AUROC
  macro_f1_vs_ratio.png/pdf    macro-F1 随标注比例变化（误差棒 = 跨 seed 标准差）

Usage:
    python evaluation/aggregate_results.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))

from common.paths import cfg_path, load_config  # noqa: E402

RUN_PATTERN = re.compile(r"^r(?P<ratio>[0-9.]+)_s(?P<seed>\d+)$")
METHOD_ORDER = ["A", "B", "W", "R", "C", "O"]
METHOD_NAMES = {
    "A": "A supervised only",
    "B": "B confidence threshold",
    "W": "W weighted filter (no RL)",
    "R": "R random (size of C)",
    "C": "C full model (RL)",
    "O": "O oracle (gold labels)",
}
# (method, reference) pairs tested with paired bootstrap
COMPARISONS = [("B", "A"), ("W", "A"), ("R", "A"), ("C", "A"), ("O", "A"), ("C", "R"), ("C", "B"), ("C", "W")]


def load_predictions(path: Path) -> dict[str, tuple[int, int]]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out[str(row["id"])] = (int(row["gold"]), int(row["pred"]))
    return out


def paired_bootstrap(pred_a: dict, pred_b: dict, n_boot: int, rng: np.random.Generator) -> dict:
    ids = sorted(set(pred_a) & set(pred_b))
    gold = np.array([pred_a[i][0] for i in ids])
    pa = np.array([pred_a[i][1] for i in ids])
    pb = np.array([pred_b[i][1] for i in ids])
    observed = f1_score(gold, pa, average="macro", zero_division=0) - f1_score(gold, pb, average="macro", zero_division=0)
    n = len(ids)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[k] = (
            f1_score(gold[idx], pa[idx], average="macro", zero_division=0)
            - f1_score(gold[idx], pb[idx], average="macro", zero_division=0)
        )
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_test": n,
        "delta_macro_f1": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_delta_le_0": float((deltas <= 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate v2 ablation results across runs.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs_dir = cfg_path(cfg, "runs_dir")
    out_dir = cfg_path(cfg, "results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    preds: dict[tuple[str, str], Path] = {}
    quality_rows = []
    skipped: list[dict] = []
    for run in sorted(runs_dir.glob("r*_s*")):
        m = RUN_PATTERN.match(run.name)
        if not m:
            continue
        ratio, seed = float(m["ratio"]), int(m["seed"])
        for res_path in sorted((run / "outputs" / "detector").glob("*/test_results.json")):
            with open(res_path, encoding="utf-8") as fh:
                res = json.load(fh)
            rows.append({"run": run.name, "ratio": ratio, "seed": seed, **res})
            pred_path = res_path.parent / "test_predictions.jsonl"
            if pred_path.exists():
                preds[(run.name, res["method"])] = pred_path
        for skip_path in sorted((run / "outputs" / "detector").glob("*/skipped.json")):
            if not (skip_path.parent / "test_results.json").exists():
                with open(skip_path, encoding="utf-8") as fh:
                    skipped.append(json.load(fh))
        q_path = run / "outputs" / "pseudo_label_quality.json"
        if q_path.exists():
            with open(q_path, encoding="utf-8") as fh:
                q = json.load(fh)
            row = {"run": run.name, "ratio": ratio, "seed": seed,
                   "pool_acc": q["full_pool"].get("accuracy"),
                   **{f"set_{k}_acc": v.get("accuracy") for k, v in q["methods"].items()},
                   **{f"set_{k}_size": v.get("size") for k, v in q["methods"].items()},
                   **{f"auroc_{k}": v for k, v in q.get("signal_auroc_for_correctness", {}).items()
                      if isinstance(v, float)}}
            quality_rows.append(row)

    if not rows:
        raise SystemExit(f"No detector results found under {runs_dir}")

    # ---- mean ± std per (ratio, method) ----
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["ratio"], r["method"])].append(r)
    summary = []
    for (ratio, method), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], METHOD_ORDER.index(kv[0][1]))):
        entry = {"ratio": ratio, "method": method, "name": METHOD_NAMES[method], "n_seeds": len(items)}
        for metric in ("accuracy", "macro_f1", "auc"):
            vals = np.array([float(i[metric]) for i in items])
            entry[f"{metric}_mean"] = float(vals.mean())
            entry[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        entry["pseudo_size_mean"] = float(np.mean([i.get("pseudo_size", 0) for i in items]))
        summary.append(entry)

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    lines = ["| label ratio | method | seeds | accuracy | macro-F1 | AUC | pseudo size |", "|---|---|---|---|---|---|---|"]
    for e in summary:
        lines.append(
            f"| {e['ratio']:.2f} | {e['name']} | {e['n_seeds']} | "
            f"{e['accuracy_mean']:.4f} ± {e['accuracy_std']:.4f} | "
            f"{e['macro_f1_mean']:.4f} ± {e['macro_f1_std']:.4f} | "
            f"{e['auc_mean']:.4f} ± {e['auc_std']:.4f} | {e['pseudo_size_mean']:.0f} |"
        )
    if skipped:
        lines += ["", "Skipped (empty pseudo set):", ""]
        lines += [f"- {s['run']} method {s['method']}: {s['reason']}" for s in skipped]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- paired bootstrap per run ----
    rng = np.random.default_rng(0)
    sig_rows = []
    for run_name in sorted({r["run"] for r in rows}):
        m = RUN_PATTERN.match(run_name)
        for method, ref in COMPARISONS:
            if (run_name, method) in preds and (run_name, ref) in preds:
                res = paired_bootstrap(
                    load_predictions(preds[(run_name, method)]),
                    load_predictions(preds[(run_name, ref)]),
                    args.n_boot,
                    rng,
                )
                sig_rows.append({"run": run_name, "ratio": float(m["ratio"]), "seed": int(m["seed"]),
                                 "comparison": f"{method} vs {ref}", **res,
                                 "significant_95": res["ci_low"] > 0 or res["ci_high"] < 0})
    if sig_rows:
        with open(out_dir / "significance.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(sig_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sig_rows)

    if quality_rows:
        fields = sorted({k for r in quality_rows for k in r}, key=lambda k: (k not in ("run", "ratio", "seed"), k))
        with open(out_dir / "pseudo_quality.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(quality_rows)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHOD_ORDER:
        pts = [e for e in summary if e["method"] == method]
        if not pts:
            continue
        ax.errorbar(
            [e["ratio"] for e in pts],
            [e["macro_f1_mean"] for e in pts],
            yerr=[e["macro_f1_std"] for e in pts],
            marker="o", capsize=3, label=METHOD_NAMES[method],
            linestyle="--" if method == "O" else "-",
        )
    ax.set_xlabel("Labeled fraction of train")
    ax.set_ylabel("Test macro-F1 (mean ± std over seeds)")
    ax.set_title("Ablation: macro-F1 vs. label ratio")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "macro_f1_vs_ratio.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "macro_f1_vs_ratio.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
