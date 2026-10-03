"""
build_baseline_sets.py
======================
为消融实验的各个对照组生成伪标签集合（全部为 claim + evidence 句对）：

    B  置信度阈值：pseudo_pool 中 confidence >= threshold，权重统一为 1.0（同 v1 Baseline B）
    L  B+L：在 B 的基础上再要求方向一致性 c > consistency_threshold，权重 1.0
       —— L 与 B 只差 LogicScore 一个条件，直接检验 LogicScore 是否有用（README 7.3 第二档）
    Q  L-q：在 B 的基础上，SUPPORTS / REFUTES 各自在类内按方向一致性 c 保留前 consistency_quantile，
       NEI 不用 c（c 对 NEI 没有区分能力，README 8.6），权重 1.0
    K  同规模置信度对照：按置信度取前 |Q| 条，权重 1.0
       —— Q 与 K 规模相同、只差选样规则，把"挑得更准"和"样本更少"分开
    W  复合权重过滤（无 RL）：pseudo_filtered 全部样本，保留复合权重
    R  随机子集：从 pseudo_filtered 中随机抽取与 C（RL 选择）同样多的样本，保留复合权重
       —— C 与 R 只差"选哪些样本"，用来检验 RL 策略是否优于随机
    O  Oracle：整个无标签池使用金标签，权重 1.0 —— 半监督方法能达到的上界

C（RL 选择）由 train_rl_selector.py 生成，A 不使用伪标签。
R 依赖 C 的规模，因此须在 train_rl_selector.py 之后运行；C 缺失时跳过 R。

Usage:
    python training/build_baseline_sets.py --run_dir runs/r0.10_s42 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))

from common.data_utils import ID2LABEL, LABEL2ID, direction_consistency, label_to_id, load_jsonl, save_jsonl  # noqa: E402
from common.paths import RunPaths, load_config  # noqa: E402

KEEP_FIELDS = ("id", "claim", "evidence", "pseudo_label", "weight", "confidence", "logic_score", "source")


def slim(rec: dict, **overrides) -> dict:
    out = {k: rec.get(k) for k in KEEP_FIELDS if k in rec}
    out.update(overrides)
    return out


def label_dist(records: list[dict]) -> dict:
    return {ID2LABEL.get(k, str(k)): v for k, v in sorted(Counter(r["pseudo_label"] for r in records).items())}


def select_quantile(pool: list[dict], conf_threshold: float, quantile: float) -> list[dict]:
    """方法 Q：置信度 >= 阈值；SUPPORTS / REFUTES 各自按 c 降序保留前 quantile，NEI 全部保留。"""
    base = [r for r in pool if float(r.get("confidence", 0.0)) >= conf_threshold]
    kept = [slim(r, weight=1.0) for r in base if label_to_id(r["pseudo_label"]) == LABEL2ID["NOT_ENOUGH_INFO"]]
    for label in (LABEL2ID["SUPPORTS"], LABEL2ID["REFUTES"]):
        scored = [(direction_consistency(r["pseudo_label"], r["logic_score"]), r) for r in base if label_to_id(r["pseudo_label"]) == label]
        scored.sort(key=lambda t: -t[0])
        kept += [slim(r, weight=1.0, consistency=round(c, 4)) for c, r in scored[: int(round(len(scored) * quantile))]]
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pseudo-label sets for ablation methods B/L/Q/K/W/R/O.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", type=str, default="B,L,Q,K,W,R,O", help="Comma-separated subset of B,L,Q,K,W,R,O.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = cfg.get("experiment", {})
    paths = RunPaths(args.run_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    stats: dict = {}

    if "B" in methods:
        threshold = float(exp_cfg.get("confidence_threshold", 0.7))
        pool = load_jsonl(paths.pseudo_pool)
        kept = [slim(r, weight=1.0) for r in pool if float(r.get("confidence", 0.0)) >= threshold]
        save_jsonl(kept, paths.pseudo_set("B"))
        stats["B"] = {"threshold": threshold, "size": len(kept), "labels": label_dist(kept)}

    if "L" in methods:
        conf_threshold = float(exp_cfg.get("confidence_threshold", 0.7))
        cons_threshold = float(exp_cfg.get("consistency_threshold", 0.3))
        pool = load_jsonl(paths.pseudo_pool)
        kept = []
        for r in pool:
            if float(r.get("confidence", 0.0)) < conf_threshold:
                continue
            c = direction_consistency(r["pseudo_label"], r["logic_score"])
            if c > cons_threshold:
                kept.append(slim(r, weight=1.0, consistency=round(c, 4)))
        save_jsonl(kept, paths.pseudo_set("L"))
        stats["L"] = {"confidence_threshold": conf_threshold, "consistency_threshold": cons_threshold,
                      "size": len(kept), "labels": label_dist(kept)}

    if "Q" in methods or "K" in methods:
        conf_threshold = float(exp_cfg.get("confidence_threshold", 0.7))
        quantile = float(exp_cfg.get("consistency_quantile", 0.5))
        pool = load_jsonl(paths.pseudo_pool)
        q_set = select_quantile(pool, conf_threshold, quantile)
        if "Q" in methods:
            save_jsonl(q_set, paths.pseudo_set("Q"))
            stats["Q"] = {"confidence_threshold": conf_threshold, "consistency_quantile": quantile,
                          "size": len(q_set), "labels": label_dist(q_set)}
        if "K" in methods:
            ranked = sorted(pool, key=lambda r: -float(r.get("confidence", 0.0)))
            kept = [slim(r, weight=1.0) for r in ranked[: len(q_set)]]
            save_jsonl(kept, paths.pseudo_set("K"))
            stats["K"] = {"size": len(kept), "matched_to_Q": len(q_set),
                          "min_confidence": round(float(kept[-1]["confidence"]), 4) if kept else None,
                          "labels": label_dist(kept)}

    if "W" in methods:
        filtered = load_jsonl(paths.pseudo_filtered)
        kept = [slim(r) for r in filtered]
        save_jsonl(kept, paths.pseudo_set("W"))
        stats["W"] = {"size": len(kept), "labels": label_dist(kept)}

    if "R" in methods:
        if not paths.rl_selected.exists():
            print(f"[build_baseline_sets] {paths.rl_selected} not found; skipping R (run the RL selector first).")
        else:
            n_rl = len(load_jsonl(paths.rl_selected))
            filtered = load_jsonl(paths.pseudo_filtered)
            rng = random.Random(args.seed)
            kept = [slim(r) for r in rng.sample(filtered, min(n_rl, len(filtered)))]
            save_jsonl(kept, paths.pseudo_set("R"))
            stats["R"] = {"size": len(kept), "matched_to_C": n_rl, "labels": label_dist(kept)}

    if "O" in methods:
        gold = {g["id"]: LABEL2ID[g["label"]] for g in load_jsonl(paths.unlabeled_gold)}
        pool = load_jsonl(paths.unlabeled_pool)
        kept = [
            {
                "id": r["id"],
                "claim": r["claim"],
                "evidence": r["evidence"],
                "pseudo_label": gold[r["id"]],
                "weight": 1.0,
                "source": r.get("source", "unknown"),
            }
            for r in pool
        ]
        save_jsonl(kept, paths.pseudo_set("O"))
        stats["O"] = {"size": len(kept), "labels": label_dist(kept)}

    out = paths.pseudo / "baseline_sets_stats.json"
    if out.exists():  # 只重建部分组时保留其他组的统计
        with open(out, encoding="utf-8") as fh:
            stats = {**json.load(fh), **stats}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
