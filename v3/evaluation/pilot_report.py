"""
pilot_report.py — 第一阶段试跑的通过标准（README_v3 第 5.2 节）
==============================================================
在 seed 42 上跑完 A、B（各两个训练 seed）和 O 之后运行。全部通过才进入主实验：

  C1 收敛：所有 detector 的最佳验证点不在最后 10% 的步数内
  C2 训练噪声：换训练 seed 重跑，A 和 B 的测试 macro-F1 变化都 < 0.02
  C3 A 的最佳验证 macro-F1 ≥ extractor 的最佳验证 macro-F1（A 是最强的纯监督基线）
  C4 O − A ≥ 0.10（测试 macro-F1，仍有足够的提升空间）
  C5 没有类别塌缩：每个 detector 在测试集上每类的预测数 ≥ 该类金标签数的 50%

退出码：全部通过为 0，否则为 1。结果同时写入 outputs/pilot_report.json。

Usage:
    python evaluation/pilot_report.py --repeat_seed 1042            # 默认：配置的 runs 目录下的 r0.10_s42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import ID2LABEL, NUM_LABELS  # noqa: E402
from common.paths import RunPaths, load_config, run_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pass/fail report for the v3 pilot run.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_dir", type=str, default=None, help="Default: <runs_dir>/r<ratio>_s<seed> from the config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--repeat_seed", type=int, default=1042, help="Training seed of the repeated A/B runs.")
    parser.add_argument("--late_fraction", type=float, default=0.1)
    parser.add_argument("--max_noise", type=float, default=0.02)
    parser.add_argument("--min_headroom", type=float, default=0.10)
    parser.add_argument("--collapse_ratio", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = RunPaths(args.run_dir or run_dir(cfg, args.ratio, args.seed))
    res = paths.detector_results()
    rows: list[dict] = []

    def add(cid: str, ok: bool | None, detail: str) -> None:
        rows.append({"id": cid, "status": "PASS" if ok else ("FAIL" if ok is False else "MISSING"), "detail": detail})

    # C1 convergence
    if res:
        late = {t: f"{r['best_step']}/{r['total_optimizer_steps']}" for t, r in res.items()
                if r.get("best_step", 0) > (1 - args.late_fraction) * r.get("total_optimizer_steps", 0)}
        add("C1", not late, "全部收敛" if not late else f"最佳点在最后 {args.late_fraction:.0%}：{late}")
    else:
        add("C1", None, "没有 detector 结果")

    # C2 training noise
    rep = f"_t{args.repeat_seed}"
    diffs, missing = {}, []
    for m in ("A", "B"):
        if m in res and m + rep in res:
            diffs[m] = abs(res[m]["macro_f1"] - res[m + rep]["macro_f1"])
        else:
            missing.append(m)
    if missing:
        add("C2", None, f"缺少重复训练的结果：{missing}（需要 {', '.join(m + rep for m in missing)}）")
    else:
        ok = all(d < args.max_noise for d in diffs.values())
        add("C2", ok, " / ".join(f"{m}: |Δ| = {d:.4f}" for m, d in diffs.items()) + f"（阈值 {args.max_noise}）")

    # C3 A vs extractor on dev
    ext_best = None
    if paths.extractor_metrics.exists():
        with open(paths.extractor_metrics, encoding="utf-8") as fh:
            ext_best = json.load(fh).get("best_val_macro_f1")
    if "A" in res and ext_best is not None:
        a_val = res["A"]["best_val"]["macro_f1"]
        add("C3", a_val >= ext_best, f"A 最佳验证 F1 {a_val:.4f} vs extractor {ext_best:.4f}")
    else:
        add("C3", None, "缺少 A 的结果或 extractor 的训练记录")

    # C4 headroom
    if "A" in res and "O" in res:
        gap = res["O"]["macro_f1"] - res["A"]["macro_f1"]
        add("C4", gap >= args.min_headroom, f"O − A = {gap:+.4f}（阈值 {args.min_headroom}）")
    else:
        add("C4", None, "缺少 A 或 O 的结果")

    # C5 collapse
    if res:
        low = {}
        for t, r in res.items():
            gold, pred = r.get("test_gold_counts"), r.get("test_pred_counts")
            if gold is None or pred is None:
                continue
            bad = [f"{ID2LABEL[k]} {pred[k]}/{gold[k]}" for k in range(NUM_LABELS) if gold[k] and pred[k] < args.collapse_ratio * gold[k]]
            if bad:
                low[t] = bad
        add("C5", not low, "没有类别塌缩" if not low else f"预测/金标签：{low}")
    else:
        add("C5", None, "没有 detector 结果")

    print(f"== 试跑通过标准（{paths.root.name}）==")
    print("| 编号 | 结果 | 说明 |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r['id']} | {r['status']} | {r['detail']} |")
    print("\n| 组 | 测试 macro-F1 | 最佳步数 / 总步数 | 最佳验证 F1 |")
    print("|---|---|---|---|")
    for t, r in sorted(res.items()):
        print(f"| {t} | {r['macro_f1']:.4f} | {r.get('best_step')} / {r.get('total_optimizer_steps')} | {r['best_val']['macro_f1']:.4f} |")

    passed = all(r["status"] == "PASS" for r in rows)
    print("\n结论：" + ("全部通过，可以进入主实验。" if passed else "未通过，先按 README_v3 第 5.2 节调整再试跑。"))
    paths.outputs.mkdir(parents=True, exist_ok=True)
    with open(paths.outputs / "pilot_report.json", "w", encoding="utf-8") as fh:
        json.dump({"run": paths.root.name, "passed": passed, "criteria": rows}, fh, indent=2, ensure_ascii=False)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
