"""
aggregate_results.py (v3)
=========================
汇总 runs 目录下的全部结果（README_v3 第 6、7 节），输出到 results/：

  summary.md          总表：各组 mean ± std、预先确定的检验 H1–H3、训练噪声、伪标签质量、自动检查的警告
  summary.csv         各 (ratio, 组) 的 mean ± std：整体 / 各来源 / 来源平均 macro-F1、accuracy、AUC
  per_seed.csv        每个 run、每个 detector（含重复训练）的指标
  hypotheses.csv      每个检验：逐 seed Δ、均值、标准差、为正的 seed 数、分层 bootstrap 95% CI、结论
  training_noise.csv  同一划分换训练 seed 的测试 macro-F1 差异
  pseudo_quality.csv  各 run 的伪标签质量（来自 outputs/pseudo_label_quality.json）
  macro_f1_vs_ratio.png

统计方法（v2 只有每个 seed 内的配对 bootstrap，只反映测试样本的抽样误差）：
  - 分层 bootstrap：每次先有放回地重抽 seed，再在每个抽到的 seed 内对测试样本做配对重抽，
    计算两组 macro-F1 之差的 seed 平均值；重复 n_boot 次取 2.5% / 97.5% 分位数。
  - 判据（结果出来之前确定）：95% CI 下界 > 0，且至少 ⌈0.8·n⌉ 个 seed 的 Δ > 0，才算"成立"；
    上界 < 0 且至少 ⌈0.8·n⌉ 个 seed 的 Δ < 0 为"显著更差"；其余为"不成立"。seed 少于 2 个时不下结论。
  - 主指标为整体 macro-F1；ClimateCheck 占测试集 78%，所以同时报告三个来源 macro-F1 的平均值作为次要指标。

只用主训练 seed（训练 seed 等于划分 seed，目录名为方法名本身）的结果做比较；<方法>_t<seed> 的重复训练
只进入训练噪声表。不同配置指纹的结果混在一起时报错（--allow_mixed 可跳过）。

Usage:
    python evaluation/aggregate_results.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import load_jsonl  # noqa: E402
from common.paths import ALL_METHODS, RunPaths, cfg_path, labeled_split_path, load_config, parse_detector_tag  # noqa: E402

RUN_PATTERN = re.compile(r"^r(?P<ratio>[0-9.]+)_s(?P<seed>\d+)$")
METHOD_NAMES = {
    "A": "A 纯监督",
    "B": "B 置信度前 30%",
    "Q": "Q 置信度 + 类内 c 分位（L-q）",
    "K": "K 同规模置信度对照",
    "W": "W 复合权重（v2 设计）",
    "R": "R 随机（v2 设计）",
    "C": "C RL（v2 设计）",
    "O": "O 金标签上界",
}
# Pre-registered comparisons (README_v3 section 4).
HYPOTHESES = [
    ("H1", "B", "A", "半监督（伪标签）在最强的纯监督基线之上是否有效"),
    ("H2", "Q", "K", "同样规模下，LogicScore（方向一致性 c）能否选出更有用的伪标签（主要检验）"),
    ("H3", "Q", "B", "用 c 过滤（变小但更准）是否优于不过滤（次要，混有规模差异）"),
    ("—", "O", "A", "上界：半监督最多能提升多少"),
]
SOURCES = ("climatecheck", "climate_fever", "scifact")
SOURCE_NAMES = {"climatecheck": "ClimateCheck", "climate_fever": "Climate-FEVER", "scifact": "SciFact"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_f1(gold: np.ndarray, pred: np.ndarray, k: int = 3) -> float:
    """Macro-F1 over the classes present in gold or pred (same as sklearn's default, zero_division=0)."""
    cm = np.bincount(gold * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1) + cm.sum(0)
    present = support > 0
    if not present.any():
        return 0.0
    f1 = np.divide(2 * tp, support, out=np.zeros(k), where=support > 0)
    return float(f1[present].mean())


def source_avg_f1(gold: np.ndarray, pred: np.ndarray, src: np.ndarray) -> float:
    vals = [macro_f1(gold[src == s], pred[src == s]) for s in range(len(SOURCES)) if (src == s).any()]
    return float(np.mean(vals)) if vals else float("nan")


def overall_f1(gold: np.ndarray, pred: np.ndarray, src: np.ndarray) -> float:  # noqa: ARG001 (uniform signature)
    return macro_f1(gold, pred)


def hierarchical_bootstrap(items: list[tuple], stat, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """items: per seed (gold, pred_a, pred_b, src). Returns the 95% CI of the seed-mean of stat(a) - stat(b)."""
    n = len(items)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        total = 0.0
        for s in rng.integers(0, n, n):
            g, pa, pb, src = items[s]
            idx = rng.integers(0, len(g), len(g))
            total += stat(g[idx], pa[idx], src[idx]) - stat(g[idx], pb[idx], src[idx])
        deltas[b] = total / n
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def verdict(deltas: list[float], lo: float, hi: float) -> str:
    n = len(deltas)
    if n < 2:
        return "seed 不足，不下结论"
    need = math.ceil(0.8 * n)
    n_pos = sum(d > 0 for d in deltas)
    n_neg = sum(d < 0 for d in deltas)
    if lo > 0 and n_pos >= need:
        return "成立"
    if hi < 0 and n_neg >= need:
        return "显著更差"
    return "不成立（不显著或方向不一致）"


def fmt_ms(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate v3 ablation results across runs.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--allow_mixed", action="store_true", help="Do not stop when results come from different configs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs_dir = cfg_path(cfg, "runs_dir")
    out_dir = cfg_path(cfg, "results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    test = load_jsonl(labeled_split_path(cfg, "test"))
    test_ids = [str(r["id"]) for r in test]
    id_pos = {i: n for n, i in enumerate(test_ids)}
    src_idx = np.array([SOURCES.index(r["source"]) if r.get("source") in SOURCES else -1 for r in test])

    gold_ref: np.ndarray | None = None
    preds: dict[tuple, np.ndarray] = {}           # (ratio, seed, tag) -> predictions aligned with test_ids
    per_seed_rows: list[dict] = []
    fingerprints: dict[str, list[str]] = defaultdict(list)
    sanity_warnings: list[str] = []
    quality_rows: list[dict] = []

    for run in sorted(runs_dir.glob("r*_s*")):
        m = RUN_PATTERN.match(run.name)
        if not m:
            continue
        ratio, seed = float(m["ratio"]), int(m["seed"])
        paths = RunPaths(run)
        for tag, res in paths.detector_results().items():
            method, train_seed = parse_detector_tag(tag)
            pred_path = paths.detector_outputs(tag) / "test_predictions.jsonl"
            if not pred_path.exists():
                continue
            gold = np.full(len(test_ids), -1)
            pred = np.full(len(test_ids), -1)
            for row in load_jsonl(pred_path):
                pos = id_pos.get(str(row["id"]))
                if pos is not None:
                    gold[pos], pred[pos] = int(row["gold"]), int(row["pred"])
            if (pred < 0).any():
                raise SystemExit(f"{pred_path} does not cover the whole test set ({int((pred < 0).sum())} missing)")
            if gold_ref is None:
                gold_ref = gold
            elif not np.array_equal(gold_ref, gold):
                raise SystemExit(f"{pred_path}: gold labels differ from other runs (different test set?)")
            preds[(ratio, seed, tag)] = pred
            fingerprints[res.get("config_fingerprint") or "missing"].append(f"{run.name}/{tag}")
            row = {
                "run": run.name, "ratio": ratio, "seed": seed, "tag": tag, "method": method,
                "train_seed": res.get("train_seed", seed), "primary": tag == method,
                "pseudo_size": res.get("pseudo_size", 0), "steps": res.get("total_optimizer_steps"),
                "best_step": res.get("best_step"), "best_val_macro_f1": (res.get("best_val") or {}).get("macro_f1"),
                "accuracy": res.get("accuracy"), "auc": res.get("auc"),
                "macro_f1": macro_f1(gold, pred), "source_avg_macro_f1": source_avg_f1(gold, pred, src_idx),
            }
            for s, name in enumerate(SOURCES):
                row[f"macro_f1_{name}"] = macro_f1(gold[src_idx == s], pred[src_idx == s])
            per_seed_rows.append(row)
        if paths.sanity_report.exists():
            with open(paths.sanity_report, encoding="utf-8") as fh:
                sanity_warnings += [f"{run.name}: {w}" for w in json.load(fh).get("warnings", [])]
        if paths.quality_report.exists():
            with open(paths.quality_report, encoding="utf-8") as fh:
                q = json.load(fh)
            qrow = {"run": run.name, "ratio": ratio, "seed": seed, "pool_acc": q["full_pool"].get("accuracy")}
            for k, v in q.get("methods", {}).items():
                qrow[f"set_{k}_size"] = v.get("size")
                qrow[f"set_{k}_acc"] = v.get("accuracy")
                qrow[f"set_{k}_balanced"] = v.get("balanced_precision")
            for k, v in q.get("signal_auroc_for_correctness", {}).items():
                if isinstance(v, float):
                    qrow[f"auroc_{k}"] = v
            quality_rows.append(qrow)

    if not per_seed_rows:
        raise SystemExit(f"No detector results found under {runs_dir}")
    if len(fingerprints) > 1 and not args.allow_mixed:
        lines = "\n".join(f"  {fp}: {len(v)} 个结果，例如 {v[:3]}" for fp, v in fingerprints.items())
        raise SystemExit("结果来自不同的配置（指纹不同），不能一起汇总：\n" + lines +
                         "\n请把不同配置的结果放在不同的 runs 目录；确实要混合时加 --allow_mixed。")

    # ---- summary per (ratio, method), primary training seed only ----
    primary = [r for r in per_seed_rows if r["primary"]]
    grouped = defaultdict(list)
    for r in primary:
        grouped[(r["ratio"], r["method"])].append(r)
    summary_rows = []
    for (ratio, method), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], ALL_METHODS.index(kv[0][1]))):
        entry = {"ratio": ratio, "method": method, "name": METHOD_NAMES.get(method, method), "n_seeds": len(items),
                 "seeds": ",".join(str(i["seed"]) for i in sorted(items, key=lambda x: x["seed"])),
                 "pseudo_size_mean": float(np.mean([i["pseudo_size"] for i in items]))}
        for key in ["macro_f1", "source_avg_macro_f1", *[f"macro_f1_{s}" for s in SOURCES], "accuracy", "auc"]:
            vals = [float(i[key]) for i in items if i.get(key) is not None]
            entry[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
            entry[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            entry[f"_{key}_vals"] = vals
        summary_rows.append(entry)

    # ---- pre-registered hypotheses ----
    rng = np.random.default_rng(0)
    hyp_rows = []
    for ratio in sorted({r["ratio"] for r in primary}):
        for hid, a, b, question in HYPOTHESES:
            seeds = sorted(s for (rt, s, t) in preds if rt == ratio and t == a and (ratio, s, b) in preds)
            if not seeds:
                continue
            items = [(gold_ref, preds[(ratio, s, a)], preds[(ratio, s, b)], src_idx) for s in seeds]
            d_all = [overall_f1(g, pa, sr) - overall_f1(g, pb, sr) for g, pa, pb, sr in items]
            d_src = [source_avg_f1(g, pa, sr) - source_avg_f1(g, pb, sr) for g, pa, pb, sr in items]
            lo, hi = hierarchical_bootstrap(items, overall_f1, args.n_boot, rng)
            slo, shi = hierarchical_bootstrap(items, source_avg_f1, args.n_boot, rng)
            hyp_rows.append({
                "ratio": ratio, "id": hid, "comparison": f"{a} − {b}", "question": question, "n_seeds": len(seeds),
                "seeds": ",".join(map(str, seeds)),
                "per_seed_delta": ",".join(f"{d:+.4f}" for d in d_all),
                "mean_delta": float(np.mean(d_all)), "std_delta": float(np.std(d_all, ddof=1)) if len(d_all) > 1 else 0.0,
                "n_positive": int(sum(d > 0 for d in d_all)),
                "ci_low": lo, "ci_high": hi,
                "src_avg_mean_delta": float(np.mean(d_src)), "src_avg_ci_low": slo, "src_avg_ci_high": shi,
                "verdict": verdict(d_all, lo, hi),
                "src_avg_verdict": verdict(d_src, slo, shi),
            })

    # ---- training noise (same split, different training seed) ----
    by_key = {(r["ratio"], r["seed"], r["tag"]): r for r in per_seed_rows}
    noise_rows = []
    for r in per_seed_rows:
        if r["primary"]:
            continue
        base = by_key.get((r["ratio"], r["seed"], r["method"]))
        if base is not None:
            noise_rows.append({"run": r["run"], "method": r["method"], "train_seed": r["train_seed"],
                               "macro_f1_primary": base["macro_f1"], "macro_f1_repeat": r["macro_f1"],
                               "abs_diff": abs(base["macro_f1"] - r["macro_f1"])})

    # ---- write files ----
    write_csv(out_dir / "per_seed.csv", per_seed_rows)
    write_csv(out_dir / "summary.csv", [{k: v for k, v in e.items() if not k.startswith("_")} for e in summary_rows])
    write_csv(out_dir / "hypotheses.csv", hyp_rows)
    write_csv(out_dir / "training_noise.csv", noise_rows)
    write_csv(out_dir / "pseudo_quality.csv", quality_rows)

    fp_list = ", ".join(fingerprints)
    lines = [
        "# v3 消融实验汇总",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}；runs 目录：`{runs_dir}`；配置指纹：{fp_list}",
        "- 只用主训练 seed 的结果做比较；判据见 README_v3 第 4 节。",
        "",
        "## 1. 各组测试集 macro-F1（mean ± std）",
        "",
        "| ratio | 组 | seeds | 伪标签条数 | macro-F1 | 来源平均 macro-F1 | " + " | ".join(SOURCE_NAMES[s] for s in SOURCES) + " | accuracy | AUC |",
        "|---|---|---|---|---|---|" + "---|" * len(SOURCES) + "---|---|",
    ]
    for e in summary_rows:
        lines.append(
            f"| {e['ratio']:.2f} | {e['name']} | {e['n_seeds']} | {e['pseudo_size_mean']:.0f} | "
            f"{fmt_ms(e['_macro_f1_vals'])} | {fmt_ms(e['_source_avg_macro_f1_vals'])} | "
            + " | ".join(fmt_ms(e[f"_macro_f1_{s}_vals"]) for s in SOURCES)
            + f" | {fmt_ms(e['_accuracy_vals'])} | {fmt_ms(e['_auc_vals'])} |"
        )
    lines += ["", "## 2. 预先确定的检验（分层 bootstrap，{} 次）".format(args.n_boot), "",
              "| ratio | 编号 | 比较 | seeds | 逐 seed Δ | 均值 ± 标准差 | Δ>0 的 seed | 95% CI | 结论 | 来源平均：均值（95% CI） | 来源平均结论 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in hyp_rows:
        lines.append(
            f"| {h['ratio']:.2f} | {h['id']} | {h['comparison']} | {h['n_seeds']} | {h['per_seed_delta']} | "
            f"{h['mean_delta']:+.4f} ± {h['std_delta']:.4f} | {h['n_positive']}/{h['n_seeds']} | "
            f"({h['ci_low']:+.4f}, {h['ci_high']:+.4f}) | **{h['verdict']}** | "
            f"{h['src_avg_mean_delta']:+.4f} ({h['src_avg_ci_low']:+.4f}, {h['src_avg_ci_high']:+.4f}) | {h['src_avg_verdict']} |"
        )
    lines += ["", "判据：95% CI 下界 > 0 且至少 ⌈0.8·n⌉ 个 seed 的 Δ > 0 为\"成立\"。", ""]
    lines += ["## 3. 训练噪声（同一划分，换训练 seed）", ""]
    if noise_rows:
        lines += ["| run | 组 | 训练 seed | 主结果 | 重复 | \\|Δ\\| |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['run']} | {r['method']} | {r['train_seed']} | {r['macro_f1_primary']:.4f} | "
                  f"{r['macro_f1_repeat']:.4f} | {r['abs_diff']:.4f} |" for r in noise_rows]
    else:
        lines.append("没有重复训练的结果。")
    lines += ["", "## 4. 伪标签集合质量", ""]
    if quality_rows:
        lines += ["| run | 整池准确率 | " + " | ".join(f"{m} 条数 / 准确率 / 平衡后" for m in ("B", "Q", "K")) + " | c 的 AUROC | 置信度 AUROC |",
                  "|---|---|---|---|---|---|---|"]
        for q in quality_rows:
            cells = []
            for m in ("B", "Q", "K"):
                if q.get(f"set_{m}_size") is None:
                    cells.append("—")
                else:
                    cells.append(f"{q[f'set_{m}_size']} / {q[f'set_{m}_acc']:.3f} / {q.get(f'set_{m}_balanced') or float('nan'):.3f}")
            lines.append(f"| {q['run']} | {q['pool_acc']:.3f} | " + " | ".join(cells) +
                         f" | {q.get('auroc_direction_consistency', float('nan')):.3f} | {q.get('auroc_confidence', float('nan')):.3f} |")
    else:
        lines.append("没有 pseudo_label_quality.json。")
    lines += ["", "## 5. 自动检查的警告（sanity_check）", ""]
    lines += [f"- ⚠️ {w}" for w in sanity_warnings] or ["没有警告。"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ALL_METHODS:
        pts = [e for e in summary_rows if e["method"] == method]
        if not pts:
            continue
        ax.errorbar([e["ratio"] for e in pts], [e["macro_f1_mean"] for e in pts], yerr=[e["macro_f1_std"] for e in pts],
                    marker="o", capsize=3, label=method, linestyle="--" if method == "O" else "-")
    ax.set_xlabel("Labeled fraction of train")
    ax.set_ylabel("Test macro-F1 (mean ± std over seeds)")
    ax.set_title("v3 ablation: macro-F1 vs. label ratio")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "macro_f1_vs_ratio.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
