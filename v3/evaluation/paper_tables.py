"""
paper_tables.py
===============
论文用的表格，数字全部从汇总结果读取（不手抄）：results_fc/summary.csv、hypotheses*.csv、multiple_comparisons.csv，
分类别的预测分布与 F1 从测试预测重新计算。输出 results_fc/paper_tables/tables.md（中文表头，预览用）和
tables.tex（booktabs，英文表头）。先运行 aggregate_results.py 与 multiple_comparisons.py。

  表 1  ratio 0.1 的主要结果（A、A⊕NLI、B、F、O；整体、来源平均、分来源 macro-F1）
  表 2  ratio 0.1 的全部组：macro-F1，以及相对 A、相对 F 的差值（95% CI，判据见 README_v3 第 4 节）
  表 3  事先确定的检验与 Holm 校正
  表 4  随标注比例的变化（A、B、F、O；F − A、F − B）
  表 5  ratio 0.1 的测试集预测分布与分类别 F1

Usage:
    python evaluation/paper_tables.py --config configs/config_fc.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.paths import cfg_path, load_config  # noqa: E402
from evaluation.multiple_comparisons import load_predictions  # noqa: E402

NAMES = {  # (中文, English)
    "A": ("A 纯监督", "A: supervised only"),
    "A⊕NLI": ("A⊕NLI 测试时融合 NLI", "A⊕NLI: NLI fused at test time"),
    "AC": ("AC 只看 claim", "AC: claim only"),
    "AT": ("AT TAPT", "AT: TAPT"),
    "B": ("B extractor 伪标签", "B: extractor pseudo-labels"),
    "FIX": ("FIX FixMatch", "FIX: FixMatch"),
    "FREE": ("FREE FreeMatch", "FREE: FreeMatch"),
    "F100": ("F100 只用 NLI 的 teacher", "F100: NLI-only teacher"),
    "E": ("E 多 NLI 集成", "E: NLI ensemble"),
    "F25": ("F25 融合 teacher，α = 0.25", "F25: fused teacher, α = 0.25"),
    "F": ("**F 融合 teacher，α = 0.5**", "\\textbf{F: fused teacher, α = 0.5}"),
    "F75": ("F75 融合 teacher，α = 0.75", "F75: fused teacher, α = 0.75"),
    "O": ("O 金标签上界", "O: gold-label upper bound"),
}
MAIN = ["A", "A⊕NLI", "B", "F", "O"]
ALL = ["A", "AC", "AT", "FIX", "FREE", "B", "F100", "E", "F25", "F", "F75", "O"]
PER_CLASS = ["A", "AC", "AT", "FIX", "FREE", "B", "F100", "E", "F", "O"]
SOURCES = [("climatecheck", "ClimateCheck"), ("climate_fever", "Climate-FEVER"), ("scifact", "SciFact")]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ms(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def delta_cell(row: dict | None, flip: bool = False) -> tuple[str, str]:
    """(markdown, latex) for a mean difference with its CI; bold when it holds (or is significantly worse)."""
    if row is None:
        return "—", "--"
    m, lo, hi = float(row["mean_delta"]), float(row["ci_low"]), float(row["ci_high"])
    if flip:
        m, lo, hi = -m, -hi, -lo
    verdict = row["verdict"]
    strong = verdict == "成立" or verdict.startswith("显著更差")
    text = f"{m:+.3f} ({lo:+.3f}, {hi:+.3f})"
    return (f"**{text}**" if strong else text), (f"\\textbf{{{text}}}" if strong else text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper tables from the aggregated v3 results.")
    parser.add_argument("--config", default="configs/config_fc.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    res = cfg_path(cfg, "results_dir")
    summary = {(float(r["ratio"]), r["method"]): r for r in read_csv(res / "summary.csv")}
    tests = {}
    for name in ("hypotheses.csv", "hypotheses_13.csv", "hypotheses_14.csv"):
        for r in read_csv(res / name):
            r["verdict"] = r["verdict"].split("（")[0]
            tests.setdefault((float(r["ratio"]), r["comparison"]), r)
    holm = {(float(r["ratio"]), r["comparison"]): r for r in read_csv(res / "multiple_comparisons.csv")}
    ratios = sorted({k[0] for k in summary})
    r01 = 0.1

    md, tex = [], []

    def add_table(title_md, title_tex, header_md, header_tex, rows_md, rows_tex, note_md="", note_tex=""):
        md.extend([f"## {title_md}", "", "| " + " | ".join(header_md) + " |", "|" + "---|" * len(header_md)])
        md.extend("| " + " | ".join(r) + " |" for r in rows_md)
        md.extend(["", note_md, ""] if note_md else [""])
        tex.extend(["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{title_tex}}}",
                    "\\begin{tabular}{l" + "c" * (len(header_tex) - 1) + "}", "\\toprule",
                    " & ".join(header_tex) + " \\\\", "\\midrule"])
        tex.extend(" & ".join(r) + " \\\\" for r in rows_tex)
        tex.extend(["\\bottomrule", "\\end{tabular}"] + ([f"\\par\\smallskip\\footnotesize {note_tex}"] if note_tex else [])
                   + ["\\end{table}", ""])

    def s(ratio, m, key):
        r = summary[(ratio, m)]
        return float(r[f"{key}_mean"]), float(r[f"{key}_std"])

    # ---- Table 1 ----
    keys = ["macro_f1", "source_avg_macro_f1", *[f"macro_f1_{k}" for k, _ in SOURCES]]
    rows_md, rows_tex = [], []
    for m in MAIN:
        cells = [ms(*s(r01, m, k)) for k in keys]
        rows_md.append([NAMES[m][0], *cells])
        rows_tex.append([NAMES[m][1], *[c.replace("±", "$\\pm$") for c in cells]])
    add_table("表 1  主要结果（10% 标注，5 个 seed，测试集 macro-F1，mean ± std）",
              "Main results with 10\\% labeled data (test macro-F1, mean $\\pm$ std over 5 seeds)",
              ["组", "整体", "来源平均", *[n for _, n in SOURCES]],
              ["Method", "Overall", "Source avg.", *[n for _, n in SOURCES]], rows_md, rows_tex)

    # ---- Table 2 ----
    rows_md, rows_tex = [], []
    for m in ALL:
        mean, std = s(r01, m, "macro_f1")
        vs_a = None if m == "A" else (tests.get((r01, f"{m} − A")) or tests.get((r01, f"A − {m}")))
        flip_a = vs_a is not None and vs_a["comparison"].startswith("A − ")
        vs_f = None if m == "F" else (tests.get((r01, f"F − {m}")) or tests.get((r01, f"{m} − F")))
        flip_f = vs_f is not None and vs_f["comparison"].startswith("F − ")
        a_md, a_tex = delta_cell(vs_a, flip_a) if m != "A" else ("—", "--")
        f_md, f_tex = delta_cell(vs_f, flip_f) if m != "F" else ("—", "--")
        rows_md.append([NAMES[m][0], ms(mean, std), a_md, f_md])
        rows_tex.append([NAMES[m][1], ms(mean, std).replace("±", "$\\pm$"), a_tex, f_tex])
    add_table("表 2  全部组（10% 标注）：测试 macro-F1 与差值（均值，95% 分层 bootstrap CI）",
              "All methods with 10\\% labeled data: test macro-F1 and differences (mean, 95\\% hierarchical bootstrap CI)",
              ["组", "macro-F1", "该组 − A", "该组 − F"], ["Method", "Macro-F1", "Method $-$ A", "Method $-$ F"],
              rows_md, rows_tex,
              "加粗：按事先确定的判据成立（CI 下界 > 0 且至少 4/5 个 seed 同向），或显著更差。—：没有事先登记这项比较。",
              "Bold: significant under the pre-registered criterion (CI excludes 0 and at least 4 of 5 seeds agree in sign). "
              "--: comparison not pre-registered.")

    # ---- Table 3 ----
    rows_md, rows_tex = [], []
    for (ratio, comp), r in sorted(holm.items(), key=lambda kv: (kv[0][0] != r01, kv[0][0])):
        t = tests[(ratio, comp)]
        after = r["holds_after"] == "True"
        p = "< 0.0001" if r["p_floor"] == "True" else f"{float(r['p']):.4f}"
        row = [f"{t['id']}", f"{ratio:.2f}", comp, f"{float(t['mean_delta']):+.3f}",
               f"({float(t['ci_low']):+.3f}, {float(t['ci_high']):+.3f})", f"{t['n_positive']}/{t['n_seeds']}", p,
               f"{float(r['p_holm']):.4f}"]
        rows_md.append(row + ["**成立**" if after else "不成立"])
        rows_tex.append([c.replace("−", "$-$").replace("<", "$<$") for c in row] + ["\\textbf{yes}" if after else "no"])
    add_table("表 3  事先确定的检验（17 项）与 Holm 校正", "Pre-registered tests (17) with Holm correction",
              ["编号", "ratio", "比较", "均值 Δ", "95% CI", "Δ>0 的 seed", "p", "Holm p", "校正后"],
              ["ID", "Ratio", "Comparison", "Mean $\\Delta$", "95\\% CI", "Seeds $\\Delta>0$", "$p$", "Holm $p$", "Holds"],
              rows_md, rows_tex,
              "校正后成立：Holm 校正后 p < 0.05 且至少 4/5 个 seed 同向。与校正前相比只有 ratio 0.5 的 F1 由成立变为不成立。",
              "Holds: Holm-adjusted $p < 0.05$ and at least 4 of 5 seeds agree in sign. Only F1 at ratio 0.5 changes after correction.")

    # ---- Table 4 ----
    rows_md, rows_tex = [], []
    for ratio in ratios:
        cells = [f"{s(ratio, m, 'macro_f1')[0]:.3f}" for m in ("A", "B", "F", "O")]
        fa = delta_cell(tests.get((ratio, "F − A")))
        fb = delta_cell(tests.get((ratio, "F − B")))
        rows_md.append([f"{ratio:.0%}", *cells, fa[0], fb[0]])
        rows_tex.append([f"{ratio * 100:.0f}\\%", *cells, fa[1], fb[1]])
    add_table("表 4  随标注比例的变化（测试 macro-F1 均值；差值为均值与 95% CI）",
              "Effect of the labeled fraction (mean test macro-F1; differences with 95\\% CI)",
              ["标注比例", "A", "B", "F", "O", "F − A", "F − B"], ["Labeled", "A", "B", "F", "O", "F $-$ A", "F $-$ B"],
              rows_md, rows_tex,
              "加粗的判据同表 2。50% 时 F − A 在 Holm 校正后不再成立（表 3）。",
              "Bold as in Table 2. At 50\\%, F $-$ A no longer holds after Holm correction (Table 3).")

    # ---- Table 5 ----
    preds = load_predictions(cfg)
    gold = preds.pop("gold")
    gold_counts = np.bincount(gold, minlength=3)
    rows_md, rows_tex = [], []
    for m in PER_CLASS:
        seeds = sorted(sd for (r, sd, t) in preds if r == r01 and t == m)
        counts = np.mean([np.bincount(preds[(r01, sd, m)], minlength=3) for sd in seeds], axis=0)
        f1s = np.mean([f1_score(gold, preds[(r01, sd, m)], labels=[0, 1, 2], average=None, zero_division=0) for sd in seeds], axis=0)
        row = [" / ".join(f"{c:.0f}" for c in counts), *[f"{x:.3f}" for x in f1s]]
        rows_md.append([NAMES[m][0], *row])
        rows_tex.append([NAMES[m][1], *row])
    gold_txt = " / ".join(str(int(c)) for c in gold_counts)
    add_table(f"表 5  测试集预测分布与分类别 F1（10% 标注，5 个 seed 平均；金标签 S / R / N = {gold_txt}）",
              f"Predicted label counts and per-class F1 with 10\\% labeled data (mean over 5 seeds; gold S / R / N = {gold_txt})",
              ["组", "预测 S / R / N", "F1 SUPPORTS", "F1 REFUTES", "F1 NEI"],
              ["Method", "Predicted S / R / N", "F1 Supports", "F1 Refutes", "F1 NEI"], rows_md, rows_tex)

    out = res / "paper_tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.md").write_text("# 论文表格（由 evaluation/paper_tables.py 生成）\n\n" + "\n".join(md), encoding="utf-8")
    (out / "tables.tex").write_text("% generated by evaluation/paper_tables.py; needs \\usepackage{booktabs}\n"
                                    + "\n".join(tex).replace("α", "$\\alpha$").replace("⊕", "$\\oplus$"), encoding="utf-8")
    print(f"wrote {out / 'tables.md'} and {out / 'tables.tex'}")


if __name__ == "__main__":
    main()
