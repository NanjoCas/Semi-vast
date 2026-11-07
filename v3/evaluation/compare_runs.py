"""
compare_runs.py — 两套配置在相同划分上的配对比较（README_v3 12.4）
====================================================================

用途：runs_fc（detector 编译）沿用了 runs_f 的划分与伪标签池，两边只差 detector 的训练方式。
本脚本逐 seed 配对比较：
  1. 每个方法的测试 macro-F1（B 配置 − A 配置）；
  2. 方法之间的对比（例如 F − A）在两套配置下是否一致。

Usage:
    python evaluation/compare_runs.py --config_a configs/config_f.yaml --config_b configs/config_fc.yaml
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.paths import RunPaths, cfg_path, load_config, run_dir  # noqa: E402

CONTRASTS = [("F", "A"), ("F", "B"), ("B", "A"), ("O", "A")]


def collect(cfg: dict, ratio: float, seeds: list[int], methods: list[str]) -> dict[tuple[str, int], float]:
    out = {}
    for seed in seeds:
        results = RunPaths(run_dir(cfg, ratio, seed)).detector_results()
        for m in methods:
            if m in results:  # main training seed only (tag without _t<seed>)
                out[(m, seed)] = float(results[m]["macro_f1"])
    return out


def fmt(vals: list[float]) -> str:
    if not vals:
        return "—"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):+.4f} ± {sd:.4f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config_a", default="configs/config_f.yaml")
    p.add_argument("--config_b", default="configs/config_fc.yaml")
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--methods", default="A,B,F,O")
    p.add_argument("--out", default=None, help="Markdown output (default: <results_dir of config_b>/compare_runs.md)")
    args = p.parse_args()

    cfg_a, cfg_b = load_config(args.config_a), load_config(args.config_b)
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = [m.strip().upper() for m in args.methods.split(",")]
    res_a = collect(cfg_a, args.ratio, seeds, methods)
    res_b = collect(cfg_b, args.ratio, seeds, methods)
    name_a, name_b = Path(args.config_a).stem, Path(args.config_b).stem

    lines = [f"# 配对比较：{name_b} − {name_a}（ratio {args.ratio}）", "",
             f"- A 配置：`{args.config_a}`（{cfg_path(cfg_a, 'runs_dir').name}）；B 配置：`{args.config_b}`（{cfg_path(cfg_b, 'runs_dir').name}）",
             "- 只用主训练 seed 的结果；两边的划分与伪标签集合相同时，差异只来自训练方式。", "",
             "## 1. 各方法的测试 macro-F1", "",
             f"| 方法 | seeds | {name_a} | {name_b} | 逐 seed Δ（{name_b} − {name_a}） | Δ 均值 ± 标准差 | Δ>0 |",
             "|---|---|---|---|---|---|---|"]
    for m in methods:
        pairs = [(s, res_a[(m, s)], res_b[(m, s)]) for s in seeds if (m, s) in res_a and (m, s) in res_b]
        if not pairs:
            lines.append(f"| {m} | 0 | — | — | — | — | — |")
            continue
        d = [b - a for _, a, b in pairs]
        lines.append(f"| {m} | {len(pairs)} | {statistics.mean(a for _, a, _ in pairs):.4f} | {statistics.mean(b for _, _, b in pairs):.4f} | "
                     + ",".join(f"{x:+.4f}" for x in d) + f" | {fmt(d)} | {sum(x > 0 for x in d)}/{len(d)} |")

    lines += ["", "## 2. 方法之间的对比在两套配置下是否一致", "",
              f"| 对比 | seeds | {name_a}：逐 seed | {name_a}：均值 ± 标准差 | {name_b}：逐 seed | {name_b}：均值 ± 标准差 |",
              "|---|---|---|---|---|---|"]
    for hi, lo in CONTRASTS:
        if hi not in methods or lo not in methods:
            continue
        ok = [s for s in seeds if all(k in r for r in (res_a, res_b) for k in ((hi, s), (lo, s)))]
        if not ok:
            continue
        da = [res_a[(hi, s)] - res_a[(lo, s)] for s in ok]
        db = [res_b[(hi, s)] - res_b[(lo, s)] for s in ok]
        lines.append(f"| {hi} − {lo} | {len(ok)} | " + ",".join(f"{x:+.4f}" for x in da) + f" | {fmt(da)} | "
                     + ",".join(f"{x:+.4f}" for x in db) + f" | {fmt(db)} |")

    text = "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else cfg_path(cfg_b, "results_dir") / "compare_runs.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[compare_runs] -> {out}")


if __name__ == "__main__":
    main()
