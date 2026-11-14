"""
export_plot_data.py
===================
把画图需要、但和 claim / 证据原文存在同一个文件里的数据，导出成不含原文的版本（v3/plot_data/），随仓库推送，
在别的机器上画图时不需要原始数据集和 processed/。只写 id、来源、标签、概率与分数，不写任何文本。

  plot_data/test.jsonl                     测试集：id、source、label（0 S / 1 R / 2 N）、nli_probs（models.nli_model，[S, R, N]）
  plot_data/<runs 目录>/<run>/pool.jsonl   无标签池：id、source、gold、extractor 的 pseudo_label / confidence / probs /
                                           entropy / logic_score / discourse_score / weight，各 NLI 模型的概率 nli_probs，
                                           以及进入了哪些伪标签集合、在集合里的标签 sets（如 {"B": 0, "F": 2}）
  plot_data/backtranslation_check.json     回译检查（README_v3 14.4）

测试预测（runs*/<run>/outputs/detector/<tag>/test_predictions.jsonl）与汇总 CSV（results*/*.csv）本身不含文本，
由 .gitignore 的例外规则直接纳入仓库，不在这里复制。

Usage:
    python tools/export_plot_data.py --config configs/config_fc.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import label_to_id, load_jsonl  # noqa: E402
from common.paths import (  # noqa: E402
    ENSEMBLE_NLI_MODELS,
    RunPaths,
    backtranslation_path,
    cfg_path,
    labeled_split_path,
    load_config,
    nli_split_path,
)

OUT = ROOT / "plot_data"
POOL_KEYS = ("pseudo_label", "confidence", "probs", "entropy", "logic_score", "discourse_score", "weight")
SET_PATTERN = re.compile(r"^set_(?P<method>[A-Z][A-Za-z0-9]*)_.+\.jsonl$")


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def export_run(run: Path, cfg: dict) -> int | None:
    paths = RunPaths(run)
    if not paths.pseudo_pool.exists():
        return None
    gold = {r["id"]: label_to_id(r["label"]) for r in load_jsonl(paths.unlabeled_gold)}
    nli = {}
    for model in [cfg["models"]["nli_model"], *ENSEMBLE_NLI_MODELS]:
        p = paths.pool_nli_probs(None if model == cfg["models"]["nli_model"] else model)
        if p.exists():
            nli[model] = {r["id"]: r["probs"] for r in load_jsonl(p)}
    sets: dict[str, dict[str, int]] = {}
    for f in sorted(paths.pseudo.glob("set_*.jsonl")):
        m = SET_PATTERN.match(f.name)
        if m:
            for r in load_jsonl(f):
                sets.setdefault(r["id"], {})[m["method"]] = int(r["pseudo_label"])
    rows = []
    for r in load_jsonl(paths.pseudo_pool):
        row = {"id": r["id"], "source": r.get("source"), "gold": gold.get(r["id"])}
        row.update({k: r.get(k) for k in POOL_KEYS})
        row["nli_probs"] = {m: v.get(r["id"]) for m, v in nli.items()}
        row["sets"] = sets.get(r["id"], {})
        rows.append(row)
    write_jsonl(rows, OUT / run.parent.name / run.name / "pool.jsonl")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export text-free plotting data to v3/plot_data/.")
    parser.add_argument("--config", default="configs/config_fc.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    test = load_jsonl(labeled_split_path(cfg, "test"))
    nli = {str(r["id"]): r["probs"] for r in load_jsonl(nli_split_path(cfg, "test"))} if nli_split_path(cfg, "test").exists() else {}
    write_jsonl([{"id": str(r["id"]), "source": r.get("source"), "label": label_to_id(r["label"]),
                  "nli_probs": nli.get(str(r["id"]))} for r in test], OUT / "test.jsonl")
    print(f"test: {len(test)} pairs")
    for run in sorted(cfg_path(cfg, "runs_dir").glob("r*_s*")):
        n = export_run(run, cfg)
        if n is not None:
            print(f"{run.parent.name}/{run.name}: {n} pool pairs")
    check = backtranslation_path(cfg, "check").with_suffix(".json")
    if check.exists():
        shutil.copy(check, OUT / "backtranslation_check.json")


if __name__ == "__main__":
    main()
