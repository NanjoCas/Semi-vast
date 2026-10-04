"""
pilot_report_f.py — 方向一（融合 teacher，方法 F）试跑的通过标准（README_v3 10.7）
==================================================================================
在 seed 42 上跑完 A、F（各两个训练 seed）和 O 之后运行。通过标准（2026-10-04 与项目负责人确定）：

  G1 两个训练 seed 下，F − A 都 > 0.02（测试 macro-F1；大于训练噪声）
  G2 两个训练 seed 下，F 都优于 A⊕NLI：A 的测试概率与 NLI 概率按同一个 α 融合，不用伪标签。
     用来排除"提升只是来自 NLI 本身的知识，而不是半监督"。

只作参考、不决定是否通过：收敛（最佳点是否在最后 10%）、训练噪声、类别塌缩、O − A、
F 拿到的提升空间比例 (F − A) / (O − A)、来源平均与分来源的 macro-F1。

退出码：全部通过为 0，否则为 1。结果同时写入 outputs/pilot_report_f.json。

Usage:
    python evaluation/pilot_report_f.py --config configs/config_f.yaml --repeat_seed 1042
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import ID2LABEL, NUM_LABELS, load_jsonl  # noqa: E402
from common.paths import RunPaths, labeled_split_path, load_config, nli_fusion_alpha, nli_split_path, run_dir  # noqa: E402

SOURCES = ("climatecheck", "climate_fever", "scifact")
SOURCE_NAMES = {"climatecheck": "ClimateCheck", "climate_fever": "Climate-FEVER", "scifact": "SciFact"}


def scores(gold: np.ndarray, pred: np.ndarray, src: np.ndarray) -> dict:
    f1 = lambda g, p: float(f1_score(g, p, average="macro", labels=list(range(NUM_LABELS)), zero_division=0))
    out = {"macro_f1": f1(gold, pred)}
    for s in SOURCES:
        out[s] = f1(gold[src == s], pred[src == s])
    out["source_avg"] = float(np.mean([out[s] for s in SOURCES]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pass/fail report for the direction-1 (fused teacher) pilot.")
    parser.add_argument("--config", type=str, default="configs/config_f.yaml")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--repeat_seed", type=int, default=1042)
    parser.add_argument("--min_gain", type=float, default=0.02)
    parser.add_argument("--late_fraction", type=float, default=0.1)
    parser.add_argument("--collapse_ratio", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    alpha = nli_fusion_alpha(cfg)
    paths = RunPaths(args.run_dir or run_dir(cfg, args.ratio, args.seed))
    res = paths.detector_results()
    test = {str(r["id"]): r for r in load_jsonl(labeled_split_path(cfg, "test"))}
    nli = {str(r["id"]): r["probs"] for r in load_jsonl(nli_split_path(cfg, "test"))} if nli_split_path(cfg, "test").exists() else None

    # ---- test scores of every detector (and of A⊕NLI) from the saved predictions ----
    table: dict[str, dict] = {}
    for tag in sorted(res):
        rows = load_jsonl(paths.detector_outputs(tag) / "test_predictions.jsonl")
        gold = np.array([int(r["gold"]) for r in rows])
        src = np.array([test[str(r["id"])]["source"] for r in rows])
        sc = scores(gold, np.array([int(r["pred"]) for r in rows]), src)
        if abs(sc["macro_f1"] - res[tag]["macro_f1"]) > 1e-6:
            raise SystemExit(f"{tag}: macro-F1 from predictions {sc['macro_f1']:.6f} != test_results.json {res[tag]['macro_f1']:.6f}")
        table[tag] = sc
        if tag.split("_t")[0] == "A" and nli is not None:
            probs = (1 - alpha) * np.array([r["probs"] for r in rows], dtype=float) \
                + alpha * np.array([nli[str(r["id"])] for r in rows], dtype=float)
            table[tag.replace("A", "A⊕NLI", 1)] = scores(gold, probs.argmax(1), src)

    rep = f"_t{args.repeat_seed}"
    rows_out: list[dict] = []
    ref_out: list[dict] = []

    def add(target: list, cid: str, ok: bool | None, detail: str) -> None:
        target.append({"id": cid, "status": "PASS" if ok else ("FAIL" if ok is False else "MISSING"), "detail": detail})

    # G1 F − A > min_gain under both training seeds
    pairs = [("F", "A"), ("F" + rep, "A" + rep)]
    if all(a in table and b in table for a, b in pairs):
        d = {a: table[a]["macro_f1"] - table[b]["macro_f1"] for a, b in pairs}
        add(rows_out, "G1", all(v > args.min_gain for v in d.values()),
            " / ".join(f"{a} − {b} = {d[a]:+.4f}" for a, b in pairs) + f"（阈值 > {args.min_gain}）")
    else:
        add(rows_out, "G1", None, f"缺少结果：需要 {', '.join(t for p in pairs for t in p)}")

    # G2 F > A⊕NLI under both training seeds
    pairs = [("F", "A⊕NLI"), ("F" + rep, "A⊕NLI" + rep)]
    if nli is None:
        add(rows_out, "G2", None, f"缺少测试集的 NLI 概率 {nli_split_path(cfg, 'test')}（run_all.py 的 NLI probs (test) 步骤）")
    elif all(a in table and b in table for a, b in pairs):
        d = {a: table[a]["macro_f1"] - table[b]["macro_f1"] for a, b in pairs}
        add(rows_out, "G2", all(v > 0 for v in d.values()),
            " / ".join(f"{a} − {b} = {d[a]:+.4f}" for a, b in pairs) + f"（α = {alpha}）")
    else:
        add(rows_out, "G2", None, f"缺少结果：需要 {', '.join(t for p in pairs for t in p)}")

    # ---- reference only ----
    late = {t: f"{r['best_step']}/{r['total_optimizer_steps']}" for t, r in res.items()
            if r.get("best_step", 0) > (1 - args.late_fraction) * r.get("total_optimizer_steps", 0)}
    add(ref_out, "收敛", not late, "全部收敛" if not late else f"最佳点在最后 {args.late_fraction:.0%}：{late}")
    noise = {m: abs(table[m]["macro_f1"] - table[m + rep]["macro_f1"]) for m in ("A", "F") if m in table and m + rep in table}
    add(ref_out, "训练噪声", all(v < 0.02 for v in noise.values()) if noise else None,
        " / ".join(f"{m}: |Δ| = {v:.4f}" for m, v in noise.items()) or "缺少重复训练的结果")
    low = {}
    for t, r in res.items():
        gold_c, pred_c = r.get("test_gold_counts"), r.get("test_pred_counts")
        if gold_c and pred_c:
            bad = [f"{ID2LABEL[k]} {pred_c[k]}/{gold_c[k]}" for k in range(NUM_LABELS) if gold_c[k] and pred_c[k] < args.collapse_ratio * gold_c[k]]
            if bad:
                low[t] = bad
    add(ref_out, "类别塌缩", not low, "没有类别塌缩" if not low else f"预测/金标签：{low}")
    if "A" in table and "O" in table:
        head = table["O"]["macro_f1"] - table["A"]["macro_f1"]
        detail = f"O − A = {head:+.4f}"
        if "F" in table and head > 0:
            detail += f"；F 拿到的提升空间 (F − A) / (O − A) = {(table['F']['macro_f1'] - table['A']['macro_f1']) / head:.0%}"
        add(ref_out, "提升空间", None if head <= 0 else True, detail)

    print(f"== 方向一试跑的通过标准（{paths.root.name}，α = {alpha}）==")
    print("| 编号 | 结果 | 说明 |")
    print("|---|---|---|")
    for r in rows_out:
        print(f"| {r['id']} | {r['status']} | {r['detail']} |")
    print("\n参考（不决定是否通过）：")
    print("| 项目 | 结果 | 说明 |")
    print("|---|---|---|")
    for r in ref_out:
        status = {"PASS": "✅", "FAIL": "⚠️", "MISSING": "—"}[r["status"]]
        print(f"| {r['id']} | {status} | {r['detail']} |")
    print("\n| 组 | 测试 macro-F1 | 来源平均 | " + " | ".join(SOURCE_NAMES[s] for s in SOURCES) + " | 最佳步数 | 最佳验证 F1 |")
    print("|---|---|---|---|---|---|---|---|")
    for t, sc in table.items():
        r = res.get(t, {})
        best = f"{r['best_step']} / {r['total_optimizer_steps']}" if r else "—（不训练）"
        bval = f"{r['best_val']['macro_f1']:.4f}" if r else "—"
        print(f"| {t} | {sc['macro_f1']:.4f} | {sc['source_avg']:.4f} | " + " | ".join(f"{sc[s]:.4f}" for s in SOURCES) + f" | {best} | {bval} |")

    passed = all(r["status"] == "PASS" for r in rows_out)
    print("\n结论：" + ("全部通过，方向一可以进入主实验。" if passed else "未通过，见 README_v3 第 10 节，由项目负责人决定下一步。"))
    paths.outputs.mkdir(parents=True, exist_ok=True)
    with open(paths.outputs / "pilot_report_f.json", "w", encoding="utf-8") as fh:
        json.dump({"run": paths.root.name, "nli_fusion_alpha": alpha, "passed": passed, "criteria": rows_out,
                   "reference": ref_out, "scores": table}, fh, indent=2, ensure_ascii=False)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
