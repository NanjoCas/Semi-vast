"""
multiple_comparisons.py
=======================
事先确定的检验做多重比较校正（Holm）：README_v3 第 4 / 10.8 / 13.4 / 14.5 节的检验都没有校正，这里补上，
回答"成立的结论会不会被多重比较推翻"。只读已有的测试预测，不改动任何结果。

- 检验族：事先登记为主要或次要的检验，共 17 项——F1–F3（每个 ratio）、X1、R1–R3、E1、T1（ratio 0.1）、
  U1、U2（ratio 0.1）。参考比较（H1、O − A 与各"—"行）不在族内。
- 每项检验用与 aggregate_results.py 相同的分层 bootstrap（先重抽 seed，再在 seed 内重抽测试样本），
  主指标为整体 macro-F1，抽 n_boot 次（默认 10000，单独的随机数生成器），
  双侧 p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0))；没有一次落到另一侧时记为 < 1/n_boot。
- Holm 校正后 p < 0.05，且至少 ⌈0.8·n⌉ 个 seed 方向一致，才算校正后仍"成立"。

Usage:
    python evaluation/multiple_comparisons.py --config configs/config_fc.yaml   # -> results_fc/multiple_comparisons.md
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import load_jsonl  # noqa: E402
from common.paths import RunPaths, cfg_path, labeled_split_path, load_config, nli_split_path  # noqa: E402
from evaluation.aggregate_results import (  # noqa: E402
    HYPOTHESES,
    HYPOTHESES_13,
    HYPOTHESES_14,
    NLI_CONTROL,
    macro_f1,
    write_csv,
)

RUN_PATTERN = re.compile(r"^r(?P<ratio>[0-9.]+)_s(?P<seed>\d+)$")
FAMILY_IDS = {"F1", "F2", "F3", "X1", "R1", "R2", "R3", "E1", "T1", "U1", "U2"}


def load_predictions(cfg: dict) -> dict[tuple, np.ndarray]:
    """(ratio, seed, method) -> predictions aligned with the test set; primary training seed only, plus A⊕NLI."""
    test = load_jsonl(labeled_split_path(cfg, "test"))
    ids = [str(r["id"]) for r in test]
    pos = {i: n for n, i in enumerate(ids)}
    alpha = (cfg.get("teacher") or {}).get("nli_fusion_alpha")
    nli = None
    if alpha is not None and nli_split_path(cfg, "test").exists():
        rows = {str(r["id"]): r["probs"] for r in load_jsonl(nli_split_path(cfg, "test"))}
        nli = np.array([rows[i] for i in ids], dtype=float)
    preds, gold_ref = {}, None
    for run in sorted(cfg_path(cfg, "runs_dir").glob("r*_s*")):
        m = RUN_PATTERN.match(run.name)
        if not m:
            continue
        ratio, seed = float(m["ratio"]), int(m["seed"])
        paths = RunPaths(run)
        for tag in paths.detector_results():
            pred_path = paths.detector_outputs(tag) / "test_predictions.jsonl"
            if "_t" in tag or not pred_path.exists():
                continue
            gold = np.full(len(ids), -1)
            pred = np.full(len(ids), -1)
            probs = np.full((len(ids), 3), np.nan)
            for row in load_jsonl(pred_path):
                k = pos[str(row["id"])]
                gold[k], pred[k] = int(row["gold"]), int(row["pred"])
                probs[k] = row["probs"]
            if gold_ref is None:
                gold_ref = gold
            elif not np.array_equal(gold, gold_ref):
                raise SystemExit(f"{pred_path}: gold labels differ from other runs")
            preds[(ratio, seed, tag)] = pred
            if tag == "A" and nli is not None:
                preds[(ratio, seed, NLI_CONTROL)] = ((1 - float(alpha)) * probs + float(alpha) * nli).argmax(1)
    preds["gold"] = gold_ref
    return preds


def bootstrap_deltas(gold: np.ndarray, pairs: list[tuple[np.ndarray, np.ndarray]], n_boot: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Hierarchical bootstrap of the seed-mean of macro-F1(a) - macro-F1(b), as in aggregate_results.py."""
    n, m = len(pairs), len(gold)
    out = np.empty(n_boot)
    for b in range(n_boot):
        total = 0.0
        for s in rng.integers(0, n, n):
            idx = rng.integers(0, m, m)
            pa, pb = pairs[s]
            total += macro_f1(gold[idx], pa[idx]) - macro_f1(gold[idx], pb[idx])
        out[b] = total / n
    return out


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Holm correction of the pre-registered tests (overall macro-F1).")
    parser.add_argument("--config", default="configs/config_fc.yaml")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    cfg = load_config(args.config)
    preds = load_predictions(cfg)
    gold = preds.pop("gold")
    ratios = sorted({k[0] for k in preds})
    family = [(r, hid, a, b) for r in ratios for hid, a, b, _ in HYPOTHESES if hid in FAMILY_IDS]
    family += [(0.1, hid, a, b) for hid, a, b, _ in [*HYPOTHESES_13, *HYPOTHESES_14] if hid in FAMILY_IDS]

    rng = np.random.default_rng(args.seed)
    rows, t0 = [], time.time()
    for ratio, hid, a, b in family:
        seeds = sorted(s for (r, s, t) in preds if r == ratio and t == a and (ratio, s, b) in preds)
        if not seeds:
            continue
        pairs = [(preds[(ratio, s, a)], preds[(ratio, s, b)]) for s in seeds]
        deltas = [macro_f1(gold, pa) - macro_f1(gold, pb) for pa, pb in pairs]
        draws = bootstrap_deltas(gold, pairs, args.n_boot, rng)
        p = 2 * min(float((draws <= 0).mean()), float((draws >= 0).mean()))
        sign = 1 if np.mean(deltas) > 0 else -1
        rows.append({"ratio": ratio, "id": hid, "comparison": f"{a} − {b}", "n_seeds": len(seeds),
                     "mean_delta": float(np.mean(deltas)), "same_direction": int(sum(np.sign(deltas) == sign)),
                     "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5)),
                     "p": max(p, 1.0 / args.n_boot), "p_floor": p == 0.0})
        print(f"  {ratio:.2f} {hid} {a} − {b}: mean {rows[-1]['mean_delta']:+.4f}, p {p:.4f} ({time.time() - t0:.0f}s)", flush=True)
    for row, adj in zip(rows, holm([r["p"] for r in rows])):
        need = math.ceil(0.8 * row["n_seeds"])
        row["p_holm"] = adj
        row["holds_before"] = row["ci_low"] > 0 and row["same_direction"] >= need and row["mean_delta"] > 0
        row["holds_after"] = adj < 0.05 and row["same_direction"] >= need and row["mean_delta"] > 0

    out_dir = cfg_path(cfg, "results_dir")
    write_csv(out_dir / "multiple_comparisons.csv", rows)

    def fmt_p(r, key):
        return f"< {1 / args.n_boot:.4f}" if r["p_floor"] and key == "p" else f"{r[key]:.4f}"

    lines = [
        "# 多重比较校正（Holm）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}；`evaluation/multiple_comparisons.py`，配置 `{args.config}`。",
        f"- 检验族：事先登记为主要或次要的 {len(rows)} 项检验；主指标为整体测试 macro-F1。",
        f"- 每项用分层 bootstrap {args.n_boot} 次（随机数种子 {args.seed}，与 summary.md 的 2000 次独立，所以 CI 与 summary.md 略有差别），"
        "双侧 p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0))。",
        "- 校正后仍成立：Holm 校正后 p < 0.05，且至少 ⌈0.8·n⌉ 个 seed 方向一致。",
        "",
        "| ratio | 编号 | 比较 | 均值 Δ | 方向一致的 seed | 95% CI | p | Holm 校正后 p | 校正前 | 校正后 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['ratio']:.2f} | {r['id']} | {r['comparison']} | {r['mean_delta']:+.4f} | "
                     f"{r['same_direction']}/{r['n_seeds']} | ({r['ci_low']:+.4f}, {r['ci_high']:+.4f}) | {fmt_p(r, 'p')} | "
                     f"{r['p_holm']:.4f} | {'成立' if r['holds_before'] else '不成立'} | {'**成立**' if r['holds_after'] else '不成立'} |")
    changed = [f"{r['ratio']:.2f} {r['id']}" for r in rows if r["holds_before"] != r["holds_after"]]
    lines += ["", "结论有变化的检验：" + ("、".join(changed) if changed else "没有。"), ""]
    (out_dir / "multiple_comparisons.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
