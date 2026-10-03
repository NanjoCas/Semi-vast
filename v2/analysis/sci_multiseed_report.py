# 新数据集多 seed 汇总（ratio 0.1，A / B / L / Q / K / O，缺的组自动跳过）：各组测试集 macro-F1 的 mean ± std（整体与分来源）、
# 每个 seed 的伪标签集合规模 / 准确率 / 平衡后准确率、每个 seed 的配对 bootstrap。
# 运行：python v2/analysis/sci_multiseed_report.py [ratio] [seeds]（任意目录；默认 0.10 42,43,44）。
"""Cross-seed summary of the A/B/L/O runs on the new datasets."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
V2 = Path("/root/autodl-tmp/Semi-vast/v2"); sys.path.insert(0, str(V2))
from evaluation.aggregate_results import load_predictions, paired_bootstrap

RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "42,43,44").split(",")]
METHODS = "ABLQKO"
SETS = "BLQK"
SRC = ("climatecheck", "climate_fever", "scifact")
ld = lambda p: [json.loads(l) for l in open(p)]
test_src = {r["id"]: r["source"] for r in ld(V2 / "processed/sci/labeled/test.jsonl")}
runs = {s: V2 / f"runs_sci/r{RATIO:.2f}_s{s}" for s in SEEDS}


def f1_by_source(pred_path: Path) -> dict:
    pr = ld(pred_path)
    out = {"all": f1_score([r["gold"] for r in pr], [r["pred"] for r in pr], average="macro")}
    for s in SRC:
        sub = [r for r in pr if test_src[r["id"]] == s]
        out[s] = f1_score([r["gold"] for r in sub], [r["pred"] for r in sub], average="macro")
    return out


scores = {m: {} for m in METHODS}   # method -> seed -> {all, source...}
info = {m: {} for m in METHODS}     # method -> seed -> test_results.json
for seed, rd in runs.items():
    for m in METHODS:
        d = rd / "outputs/detector" / m
        if (d / "test_predictions.jsonl").exists():
            scores[m][seed] = f1_by_source(d / "test_predictions.jsonl")
            info[m][seed] = json.load(open(d / "test_results.json"))

fmt = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else (f"{v[0]:.3f}" if v else "—")
print(f"== detector 测试集 macro-F1（ratio {RATIO}，seeds {SEEDS}，mean ± std）==")
print("| 组 | seeds | 伪标签条数 | 参数更新 | 全部 | ClimateCheck | Climate-FEVER | SciFact |")
print("|---|---|---|---|---|---|---|---|")
for m in METHODS:
    if not scores[m]:
        continue
    sz = [info[m][s]["pseudo_size"] for s in scores[m]]
    st = [info[m][s].get("total_optimizer_steps") or 0 for s in scores[m]]
    cols = [fmt([scores[m][s][k] for s in scores[m]]) for k in ("all", *SRC)]
    print(f"| {m} | {len(scores[m])} | {np.mean(sz):.0f} | {np.mean(st):.0f} | " + " | ".join(cols) + " |")

print("\n== 每个 seed 的 macro-F1（全部）==")
print("| seed | " + " | ".join(METHODS) + " | L − B | Q − K |")
print("|---|" + "---|" * (len(METHODS) + 2))
for seed in SEEDS:
    row = [f"{scores[m][seed]['all']:.3f}" if seed in scores[m] else "—" for m in METHODS]
    diff = lambda a, b: (f"{scores[a][seed]['all'] - scores[b][seed]['all']:+.3f}"
                         if seed in scores[a] and seed in scores[b] else "—")
    print(f"| {seed} | " + " | ".join(row) + f" | {diff('L', 'B')} | {diff('Q', 'K')} |")

print("\n== 伪标签集合质量（隐藏金标签）==")
print("| seed | 整池准确率 | " + " | ".join(f"{m} 条数 / 准确率 / 平衡后" for m in SETS) + " | c 的 AUROC | 置信度 AUROC |")
print("|---|---|" + "---|" * len(SETS) + "---|---|")
for seed, rd in runs.items():
    qp = rd / "outputs/pseudo_label_quality.json"
    if not qp.exists():
        continue
    q = json.load(open(qp)); ms = q["methods"]; au = q["signal_auroc_for_correctness"]
    cell = lambda k: (f"{ms[k]['size']} / {ms[k]['accuracy']:.3f} / {ms[k].get('balanced_precision', float('nan')):.3f}"
                      if k in ms else "—")
    print(f"| {seed} | {q['full_pool']['accuracy']:.3f} | " + " | ".join(cell(m) for m in SETS) + " | "
          f"{au.get('direction_consistency', float('nan')):.3f} | {au.get('confidence', float('nan')):.3f} |")

print("\n== 配对 bootstrap（每个 seed 单独，1000 次）==")
rng = np.random.default_rng(0)
print("| 比较 | " + " | ".join(f"seed {s}" for s in SEEDS) + " |")
print("|---|" + "---|" * len(SEEDS))
for a, b in (("B", "A"), ("L", "A"), ("L", "B"), ("Q", "A"), ("K", "A"), ("Q", "K"), ("Q", "B"), ("O", "A")):
    cells = []
    for seed, rd in runs.items():
        pa, pb = rd / f"outputs/detector/{a}/test_predictions.jsonl", rd / f"outputs/detector/{b}/test_predictions.jsonl"
        if pa.exists() and pb.exists():
            r = paired_bootstrap(load_predictions(pa), load_predictions(pb), 1000, rng)
            sig = "*" if r["ci_low"] > 0 or r["ci_high"] < 0 else ""
            cells.append(f"{r['delta_macro_f1']:+.3f}{sig} ({r['ci_low']:+.3f}, {r['ci_high']:+.3f})")
        else:
            cells.append("—")
    print(f"| {a} − {b} | " + " | ".join(cells) + " |")
print("（* 表示 95% CI 不含 0）")
