"""
build_baseline_sets.py (v3)
===========================
为消融实验的各个对照组生成伪标签集合（全部为 claim + evidence 句对，README_v3 第 4 节）：

    B  置信度前 confidence_top_fraction（默认 30%）：按置信度排序取前 30%，权重 1.0。
       v2 用固定阈值 0.7，但 extractor 没有校准，各 seed 中置信度 ≥ 0.7 的比例从 18.5% 到 68.7% 不等，
       B 的规模相差 3.7 倍；按比例选择让各 seed 的规模一致。
    Q  L-q：在 B 内，SUPPORTS / REFUTES 各自按方向一致性 c 降序保留前 consistency_quantile（默认 50%），
       NEI 全部保留（c 对 NEI 没有区分能力，见 v2 README 8.6），权重 1.0。
    K  同规模置信度对照：按置信度取前 |Q| 条（因此 K ⊂ B），权重 1.0。
       —— Q 与 K 规模相同、都在 B 内，只差选样规则：这是检验 LogicScore 的主要对比（H2）。
    F  方向一（README_v3 10.6）：logic-aware teacher。伪标签与置信度来自 extractor 与 NLI 的融合
       p = (1 − α)·p_extractor + α·p_NLI（α = teacher.nli_fusion_alpha，NLI 的蕴含 / 矛盾 / 中立对应 SUP / REF / NEI），
       按融合后的置信度取前 confidence_top_fraction，权重 1.0。需要 pseudo/nli_probs.jsonl（training/compute_nli_probs.py）。
    O  Oracle：整个无标签池使用金标签，权重 1.0 —— 半监督方法能达到的上界。

    W / R 沿用 v2 的设计（尚未按 v3 重新设计，结论不可用，见 README_v3 第 8 节）：
    W  复合权重 ≥ weight_threshold 的全部样本，保留复合权重
    R  从 W 的池子中随机抽取与 C（RL 选择）同样多的样本；C 缺失时跳过

排序在置信度（或 c）相同时按 id 打破平局，结果与样本在文件中的顺序无关。
C（RL 选择）由 train_rl_selector.py 生成，A 不使用伪标签。

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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import ID2LABEL, LABEL2ID, direction_consistency, label_to_id, load_jsonl, save_jsonl  # noqa: E402
from common.paths import RunPaths, load_config, nli_fusion_alpha  # noqa: E402

KEEP_FIELDS = ("id", "claim", "evidence", "pseudo_label", "weight", "confidence", "logic_score", "source")
BUILDABLE = ("B", "Q", "K", "F", "W", "R", "O")


def slim(rec: dict, **overrides) -> dict:
    out = {k: rec.get(k) for k in KEEP_FIELDS if k in rec}
    out.update(overrides)
    return out


def label_dist(records: list[dict]) -> dict:
    return {ID2LABEL.get(k, str(k)): v for k, v in sorted(Counter(label_to_id(r["pseudo_label"]) for r in records).items())}


def rank_by_confidence(pool: list[dict]) -> list[dict]:
    return sorted(pool, key=lambda r: (-float(r.get("confidence", 0.0)), str(r["id"])))


def select_top_fraction(pool: list[dict], fraction: float) -> list[dict]:
    """方法 B：按置信度取前 fraction。"""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"experiment.confidence_top_fraction must be in (0, 1], got {fraction}")
    n = int(round(len(pool) * fraction))
    return rank_by_confidence(pool)[:n]


def select_quantile(base: list[dict], quantile: float) -> list[dict]:
    """方法 Q：在 base（即 B）内，SUPPORTS / REFUTES 各自按 c 降序保留前 quantile，NEI 全部保留。"""
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"experiment.consistency_quantile must be in (0, 1], got {quantile}")
    kept = [r for r in base if label_to_id(r["pseudo_label"]) == LABEL2ID["NOT_ENOUGH_INFO"]]
    for label in (LABEL2ID["SUPPORTS"], LABEL2ID["REFUTES"]):
        members = [r for r in base if label_to_id(r["pseudo_label"]) == label]
        members.sort(key=lambda r: (-direction_consistency(r["pseudo_label"], r["logic_score"]), str(r["id"])))
        kept += members[: int(round(len(members) * quantile))]
    return kept


def fuse_teacher(pool: list[dict], nli: dict[str, list[float]], alpha: float) -> list[dict]:
    """方法 F：p = (1 − α)·p_extractor + α·p_NLI；伪标签 = argmax，置信度 = max。保留 extractor 原来的伪标签以便分析。"""
    missing = [r["id"] for r in pool if str(r["id"]) not in nli]
    if missing:
        raise SystemExit(f"{len(missing)} pool records have no NLI probabilities (e.g. {missing[:3]}); "
                         "run training/compute_nli_probs.py --run_dir first")
    fused = []
    for r in pool:
        p = (1.0 - alpha) * np.asarray(r["probs"], dtype=float) + alpha * np.asarray(nli[str(r["id"])], dtype=float)
        k = int(p.argmax())
        fused.append({**r, "pseudo_label": k, "confidence": float(p[k]), "fused_probs": [round(float(x), 6) for x in p],
                      "extractor_label": label_to_id(r["pseudo_label"]), "extractor_confidence": float(r["confidence"])})
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pseudo-label sets for ablation methods B/Q/K/F/W/R/O.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", type=str, default="B,Q,K,O", help="Comma-separated subset of B,Q,K,F,W,R,O.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = cfg.get("experiment", {})
    paths = RunPaths(args.run_dir)
    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(methods) - set(BUILDABLE))
    if unknown:
        raise SystemExit(f"Cannot build pseudo sets for {unknown}; buildable: {BUILDABLE}")
    stats: dict = {}

    if any(m in methods for m in ("B", "Q", "K")):
        fraction = float(exp_cfg["confidence_top_fraction"])
        quantile = float(exp_cfg.get("consistency_quantile", 0.5))
        pool = load_jsonl(paths.pseudo_pool)
        b_set = select_top_fraction(pool, fraction)
        q_set = select_quantile(b_set, quantile)
        if "B" in methods:
            kept = [slim(r, weight=1.0) for r in b_set]
            save_jsonl(kept, paths.pseudo_set("B"))
            stats["B"] = {"confidence_top_fraction": fraction, "pool_size": len(pool), "size": len(kept),
                          "min_confidence": round(float(b_set[-1]["confidence"]), 4) if b_set else None,
                          "labels": label_dist(kept)}
        if "Q" in methods:
            kept = [slim(r, weight=1.0, consistency=round(direction_consistency(r["pseudo_label"], r["logic_score"]), 4))
                    for r in q_set]
            save_jsonl(kept, paths.pseudo_set("Q"))
            stats["Q"] = {"confidence_top_fraction": fraction, "consistency_quantile": quantile,
                          "size": len(kept), "labels": label_dist(kept)}
        if "K" in methods:
            k_src = rank_by_confidence(pool)[: len(q_set)]
            kept = [slim(r, weight=1.0) for r in k_src]
            save_jsonl(kept, paths.pseudo_set("K"))
            stats["K"] = {"size": len(kept), "matched_to_Q": len(q_set),
                          "min_confidence": round(float(k_src[-1]["confidence"]), 4) if k_src else None,
                          "labels": label_dist(kept)}

    if "F" in methods:
        fraction = float(exp_cfg["confidence_top_fraction"])
        alpha = nli_fusion_alpha(cfg)
        pool = load_jsonl(paths.pseudo_pool)
        nli = {str(r["id"]): r["probs"] for r in load_jsonl(paths.nli_probs)}
        fused_pool = fuse_teacher(pool, nli, alpha)
        f_set = select_top_fraction(fused_pool, fraction)
        kept = [slim(r, weight=1.0, extractor_label=r["extractor_label"], fused_probs=r["fused_probs"]) for r in f_set]
        save_jsonl(kept, paths.pseudo_set("F"))
        stats["F"] = {"teacher": "extractor + NLI", "nli_fusion_alpha": alpha, "confidence_top_fraction": fraction,
                      "pool_size": len(pool), "size": len(kept),
                      "min_confidence": round(float(f_set[-1]["confidence"]), 4) if f_set else None,
                      "labels": label_dist(kept),
                      "pool_labels_changed_by_fusion": sum(r["pseudo_label"] != r["extractor_label"] for r in fused_pool),
                      "set_labels_changed_by_fusion": sum(r["pseudo_label"] != r["extractor_label"] for r in f_set)}

    if "W" in methods:
        filtered = load_jsonl(paths.pseudo_filtered)
        kept = [slim(r) for r in filtered]
        save_jsonl(kept, paths.pseudo_set("W"))
        stats["W"] = {"size": len(kept), "labels": label_dist(kept), "note": "v2 design, not revised in v3"}

    if "R" in methods:
        if not paths.rl_selected.exists():
            print(f"[build_baseline_sets] {paths.rl_selected} not found; skipping R (run the RL selector first).")
        else:
            n_rl = len(load_jsonl(paths.rl_selected))
            filtered = load_jsonl(paths.pseudo_filtered)
            rng = random.Random(args.seed)
            kept = [slim(r) for r in rng.sample(filtered, min(n_rl, len(filtered)))]
            save_jsonl(kept, paths.pseudo_set("R"))
            stats["R"] = {"size": len(kept), "matched_to_C": n_rl, "labels": label_dist(kept),
                          "note": "v2 design, not revised in v3"}

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

    out = paths.baseline_stats
    if out.exists():  # 只重建部分组时保留其他组的统计
        with open(out, encoding="utf-8") as fh:
            stats = {**json.load(fh), **stats}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
