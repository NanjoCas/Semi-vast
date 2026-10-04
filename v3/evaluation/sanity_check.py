"""
sanity_check.py — 每个 run 结束后的自动检查（README_v3 第 6 节）
================================================================
v2 每跑一轮都要人工翻日志才发现问题（O ≈ A、L 只预测出 39 条 REFUTES、组间训练步数不同……）。
这里把这些检查固定下来，run_all.py 在每个 run 结束后调用：

  S1 同一 run 内各 detector 的参数更新次数相同，且等于 training.detector_total_steps
  S2 最佳验证点不在最后 10% 的步数内（否则可能还没收敛）
  S3 没有类别塌缩：测试集上每类的预测数 ≥ 该类金标签数的 50%
  S4 Q 与 K 的伪标签集合规模相同
  S5 detector 结果记录的配置指纹与当前配置一致
  S6 B 的规模等于 round(confidence_top_fraction × 池大小)
  S7 detector 结果中的伪标签条数与当前的伪标签集合文件一致（集合重建后没有沿用旧结果）

只打印警告并写入 outputs/sanity_check.json，不会中断实验；aggregate_results.py 会把所有警告汇总到
results/summary.md。

Usage:
    python evaluation/sanity_check.py --run_dir runs/r0.10_s42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import ID2LABEL, NUM_LABELS  # noqa: E402
from common.fingerprint import config_fingerprint  # noqa: E402
from common.paths import RunPaths, load_config, parse_detector_tag  # noqa: E402


def _count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _counts_from_predictions(path: Path) -> tuple[list[int], list[int]] | None:
    if not path.exists():
        return None
    gold, pred = [0] * NUM_LABELS, [0] * NUM_LABELS
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                gold[int(row["gold"])] += 1
                pred[int(row["pred"])] += 1
    return gold, pred


def check_run(cfg: dict, run_dir: str | Path, late_fraction: float = 0.1, collapse_ratio: float = 0.5) -> dict:
    paths = RunPaths(run_dir)
    results = paths.detector_results()
    train_cfg = cfg.get("training", {})
    planned = int(train_cfg["detector_total_steps"]) if "detector_total_steps" in train_cfg else None
    current_fp = config_fingerprint(cfg)
    checks: list[dict] = []

    def add(cid: str, status: str, detail: str) -> None:
        checks.append({"id": cid, "status": status, "detail": detail})

    # S1 equal budgets
    if results:
        steps = {tag: r.get("total_optimizer_steps") for tag, r in results.items()}
        bad = {t: s for t, s in steps.items() if planned is not None and s != planned}
        if len(set(steps.values())) == 1 and not bad:
            add("S1", "ok", f"所有 detector 都是 {next(iter(steps.values()))} 步")
        else:
            add("S1", "warn", f"参数更新次数不一致或不等于配置的 {planned}：{steps}")
    else:
        add("S1", "skip", "还没有 detector 结果")

    # S2 convergence
    late = {}
    for tag, r in results.items():
        total = r.get("total_optimizer_steps") or 0
        best = r.get("best_step")
        if total and best is not None and best > (1.0 - late_fraction) * total:
            late[tag] = f"{best}/{total}"
    if results:
        if late:
            add("S2", "warn", f"最佳验证点在最后 {late_fraction:.0%} 的步数内，可能未收敛：{late}")
        else:
            add("S2", "ok", "所有 detector 的最佳验证点都在训练结束之前")

    # S3 class collapse
    collapsed = {}
    for tag, r in results.items():
        gold, pred = r.get("test_gold_counts"), r.get("test_pred_counts")
        if gold is None or pred is None:
            counts = _counts_from_predictions(paths.detector_outputs(tag) / "test_predictions.jsonl")
            if counts is None:
                continue
            gold, pred = counts
        low = [f"{ID2LABEL[k]} {pred[k]}/{gold[k]}" for k in range(NUM_LABELS) if gold[k] and pred[k] < collapse_ratio * gold[k]]
        if low:
            collapsed[tag] = low
    if results:
        if collapsed:
            add("S3", "warn", f"类别预测数低于金标签数的 {collapse_ratio:.0%}（预测/金标签）：{collapsed}")
        else:
            add("S3", "ok", "没有类别塌缩")

    # S4 |Q| == |K|
    nq, nk = _count_lines(paths.pseudo_set("Q")), _count_lines(paths.pseudo_set("K"))
    if nq is None or nk is None:
        add("S4", "skip", "Q 或 K 的集合不存在")
    elif nq == nk:
        add("S4", "ok", f"|Q| = |K| = {nq}")
    else:
        add("S4", "warn", f"|Q| = {nq} ≠ |K| = {nk}")

    # S5 config fingerprint
    if results:
        other = {tag: r.get("config_fingerprint") for tag, r in results.items() if r.get("config_fingerprint") != current_fp}
        if other:
            add("S5", "warn", f"这些结果来自其他配置（当前 {current_fp}）：{other}")
        else:
            add("S5", "ok", f"所有结果的配置指纹都是 {current_fp}")

    # S6 B (and F) size
    npool = _count_lines(paths.pseudo_pool)
    frac = cfg.get("experiment", {}).get("confidence_top_fraction")
    sizes = {m: _count_lines(paths.pseudo_set(m)) for m in ("B", "F")}
    sizes = {m: n for m, n in sizes.items() if n is not None}
    if not sizes or npool is None or frac is None:
        add("S6", "skip", "B / F 的集合或伪标签池不存在")
    else:
        expected = int(round(npool * float(frac)))
        bad = {m: n for m, n in sizes.items() if n != expected}
        names = " = ".join(f"|{m}|" for m in sizes)
        if bad:
            add("S6", "warn", f"round({frac} × {npool}) = {expected}，但 " + "，".join(f"|{m}| = {n}" for m, n in bad.items()))
        else:
            add("S6", "ok", f"{names} = {expected} = round({frac} × {npool})")

    # S7 detector results match the current pseudo sets
    stale = {}
    for tag, r in results.items():
        parsed = parse_detector_tag(tag)
        set_path = paths.pseudo_set(parsed[0]) if parsed is not None else None
        if set_path is None:  # method A has no pseudo set
            continue
        n_now = _count_lines(set_path)
        if n_now is not None and n_now != r.get("pseudo_size"):
            stale[tag] = f"结果 {r.get('pseudo_size')} 条，当前集合 {n_now} 条"
    if results:
        if stale:
            add("S7", "warn", f"伪标签集合在训练后被重建过，结果可能已过期：{stale}")
        else:
            add("S7", "ok", "detector 结果与当前伪标签集合一致")

    warnings = [f"{c['id']} {c['detail']}" for c in checks if c["status"] == "warn"]
    return {"run": paths.root.name, "config_fingerprint": current_fp, "checks": checks, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic checks for one run directory.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--late_fraction", type=float, default=0.1)
    parser.add_argument("--collapse_ratio", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = check_run(cfg, args.run_dir, args.late_fraction, args.collapse_ratio)
    paths = RunPaths(args.run_dir)
    paths.outputs.mkdir(parents=True, exist_ok=True)
    with open(paths.sanity_report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    icon = {"ok": "✅", "warn": "⚠️ ", "skip": "— "}
    print(f"[SANITY] {report['run']}")
    for c in report["checks"]:
        print(f"[SANITY] {icon[c['status']]} {c['id']} {c['detail']}")
    if report["warnings"]:
        print(f"[SANITY] ⚠️  {len(report['warnings'])} 项警告，详见 {paths.sanity_report}")


if __name__ == "__main__":
    main()
