"""
compare_ablation_baselines.py
==============================
Collects final test-set results for the three ablation baselines defined in
PROJECT_README.md Section 8 and produces a comparison table + bar chart:

    Baseline A: Supervised only (no pseudo-labels)
    Baseline B: Semi-supervised with confidence threshold (no RL, no LogicScore)
    Baseline C: Full model (RL selector + LogicScore/DiscourseScore weighting)

Usage:
    python evaluation/compare_ablation_baselines.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent

BASELINES = [
    {
        "key": "A",
        "label": "Baseline A\n(supervised only)",
        "results_path": PROJECT_ROOT / "outputs" / "baseline_A_supervised" / "final_test_results.json",
    },
    {
        "key": "B",
        "label": "Baseline B\n(confidence threshold)",
        "results_path": PROJECT_ROOT / "outputs" / "baseline_B_confidence" / "final_test_results.json",
    },
    {
        "key": "C",
        "label": "Baseline C\n(full model)",
        "results_path": PROJECT_ROOT / "outputs" / "final_test_results.json",
    },
]

METRICS = ["accuracy", "macro_f1", "auc"]
METRIC_LABELS = {"accuracy": "Accuracy", "macro_f1": "Macro F1", "auc": "AUC"}
COLORS = {"accuracy": "tab:blue", "macro_f1": "tab:orange", "auc": "tab:green"}


def load_results() -> list[dict]:
    rows = []
    for b in BASELINES:
        if not b["results_path"].exists():
            print(f"  [WARN] Missing results for {b['key']}: {b['results_path']}")
            rows.append({**b, "metrics": None})
            continue
        with open(b["results_path"], encoding="utf-8") as fh:
            metrics = json.load(fh)
        rows.append({**b, "metrics": metrics})
    return rows


def main() -> None:
    rows = load_results()
    available = [r for r in rows if r["metrics"] is not None]
    if not available:
        raise SystemExit(
            "No baseline results found. Run the ablation pipeline first "
            "(run_ablation_pipeline.py)."
        )

    out_dir = PROJECT_ROOT / "outputs" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Text table ----
    print("\n" + "=" * 72)
    print("Ablation Comparison — Test Set")
    print("=" * 72)
    header = f"{'Baseline':<28}{'Accuracy':>12}{'Macro F1':>12}{'AUC':>12}"
    print(header)
    print("-" * 72)
    for r in rows:
        name = r["label"].replace("\n", " ")
        if r["metrics"] is None:
            print(f"{name:<28}{'(missing)':>12}")
            continue
        m = r["metrics"]
        print(
            f"{name:<28}"
            f"{m.get('accuracy', float('nan')):>12.4f}"
            f"{m.get('macro_f1', float('nan')):>12.4f}"
            f"{m.get('auc', float('nan')):>12.4f}"
        )
    print("=" * 72)

    # ---- Save JSON summary ----
    summary = {
        r["key"]: {"label": r["label"].replace("\n", " "), **(r["metrics"] or {})}
        for r in rows
    }
    summary_path = out_dir / "ablation_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved summary to {summary_path}")

    # ---- Grouped bar chart ----
    fig, ax = plt.subplots(figsize=(9, 6))
    n_baselines = len(rows)
    n_metrics = len(METRICS)
    bar_width = 0.8 / n_metrics
    x = range(n_baselines)

    for i, metric in enumerate(METRICS):
        values = [
            (r["metrics"].get(metric, 0.0) if r["metrics"] else 0.0) for r in rows
        ]
        offsets = [xi + (i - (n_metrics - 1) / 2) * bar_width for xi in x]
        bars = ax.bar(
            offsets,
            values,
            width=bar_width,
            label=METRIC_LABELS[metric],
            color=COLORS[metric],
        )
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Comparison — Test Set (Accuracy / Macro F1 / AUC)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    png_path = out_dir / "ablation_comparison.png"
    pdf_path = out_dir / "ablation_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {png_path}")
    print(f"Saved plot to {pdf_path}")


if __name__ == "__main__":
    main()
