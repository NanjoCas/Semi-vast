"""
make_ssl_split.py
=================
把 processed/labeled/train.jsonl 按比例切成"有标签部分"和"无标签池"（标准半监督协议）。

- 按 (source, label) 分层抽样，保证有标签部分的来源与类别分布和原 train 一致；
- 无标签池保留 claim + evidence（与 dev/test 同格式），但去掉 label；
- 金标签单独写到 unlabeled_gold.jsonl，只给 evaluation/ 和 Oracle 组使用，
  训练流程（extractor / 伪标签 / RL / detector）都不读取它；
- Climate-FEVER 过采样只作用于有标签部分（v1 在 dev/test 中也做了重复）。

Usage:
    python data_prep/make_ssl_split.py --ratio 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))

from common.data_utils import load_jsonl, save_jsonl  # noqa: E402
from common.paths import RunPaths, labeled_split_path, load_config, run_dir  # noqa: E402


def stratified_split(records: list[dict], ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r.get("source", "unknown"), r["label"])].append(r)

    labeled, unlabeled = [], []
    for key in sorted(groups):
        items = groups[key][:]
        rng.shuffle(items)
        n_keep = max(1, round(len(items) * ratio))  # every (source, label) cell keeps >= 1 example
        labeled.extend(items[:n_keep])
        unlabeled.extend(items[n_keep:])
    rng.shuffle(labeled)
    rng.shuffle(unlabeled)
    return labeled, unlabeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Hide labels of part of train to build an SSL split.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--ratio", type=float, required=True, help="Fraction of train that keeps its labels.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run_dir", type=str, default=None, help="Override run directory.")
    args = parser.parse_args()

    if not 0.0 < args.ratio < 1.0:
        raise SystemExit("--ratio must be in (0, 1); ratio=1.0 leaves no unlabeled pool.")

    cfg = load_config(args.config)
    paths = RunPaths(args.run_dir or run_dir(cfg, args.ratio, args.seed))

    train_path = labeled_split_path(cfg, "train")
    if not train_path.exists():
        raise SystemExit(f"{train_path} not found. Run data_prep/build_labeled.py first.")
    records = load_jsonl(train_path)

    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{train_path} contains duplicate ids; rebuild it with build_labeled.py.")

    labeled, unlabeled = stratified_split(records, args.ratio, args.seed)

    oversample = int(cfg.get("data", {}).get("cf_oversample_labeled", 1))
    labeled_out = []
    for r in labeled:
        copies = oversample if r.get("source") == "climate_fever" and oversample > 1 else 1
        labeled_out.extend([r] * copies)
    random.Random(args.seed).shuffle(labeled_out)

    pool = [
        {"id": r["id"], "claim": r["claim"], "evidence": r["evidence"], "source": r.get("source", "unknown")}
        for r in unlabeled
    ]
    gold = [{"id": r["id"], "label": r["label"]} for r in unlabeled]

    save_jsonl(labeled_out, paths.labeled_train)
    save_jsonl(pool, paths.unlabeled_pool)
    save_jsonl(gold, paths.unlabeled_gold)

    stats = {
        "ratio": args.ratio,
        "seed": args.seed,
        "train_unique": len(records),
        "labeled_unique": len(labeled),
        "labeled_after_cf_oversample": len(labeled_out),
        "unlabeled_pool": len(pool),
        "labeled_label_dist": dict(Counter(r["label"] for r in labeled)),
        "labeled_source_dist": dict(Counter(r.get("source") for r in labeled)),
        "unlabeled_label_dist": dict(Counter(r["label"] for r in unlabeled)),
    }
    with open(paths.split_stats, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(f"[make_ssl_split] ratio={args.ratio} seed={args.seed}")
    print(f"  labeled  : {len(labeled)} unique ({len(labeled_out)} after CF x{oversample}) -> {paths.labeled_train}")
    print(f"  unlabeled: {len(pool)} -> {paths.unlabeled_pool}")
    print(f"  gold     : {paths.unlabeled_gold} (evaluation / oracle only)")


if __name__ == "__main__":
    main()
