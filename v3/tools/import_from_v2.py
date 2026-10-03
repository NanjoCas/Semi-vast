"""
import_from_v2.py — 把 v2 已有的划分和伪标签池导入 v3（README_v3 第 5.1 节）
==========================================================================

v3 没有改动数据构建、SSL 划分、extractor 和伪标签生成，所以 v2 在新数据集上（runs_sci/）
已经生成的伪标签池可以直接复用，每个 seed 省去约 10–20 分钟的 extractor 训练和伪标签生成，
而且 v3 与 v2 用的是完全相同的伪标签。导入前检查：

  1. 两边决定伪标签池的设置完全相同（common/fingerprint.pool_settings）；
  2. 有标签数据 processed/labeled/{train,dev,test}.jsonl 逐字节相同（v3 还没有时直接复制）；
  3. 每个 run 内：无标签池、金标签、伪标签池的 id 完全一致，有标签部分与无标签池不重叠，
     两者都来自 train。

只复制划分（data/）、伪标签池（pseudo/pseudo_pool.jsonl、pseudo_filtered.jsonl、pseudo_stats.json）
和 extractor 的训练记录（outputs/extractor/，试跑标准 C3 要用）。伪标签集合（B/Q/K/O）和 detector
结果全部由 v3 重新生成。可以重复运行：内容相同的文件会跳过，内容不同时报错（--overwrite 覆盖）。

注意：导入的 run 没有 extractor 权重（v2 跑完已清理），所以不能在这些 run 上跑 C / R。

Usage:
    python tools/import_from_v2.py --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import load_jsonl  # noqa: E402
from common.fingerprint import diff_settings, pool_settings  # noqa: E402
from common.paths import RunPaths, labeled_split_path, load_config, run_dir, run_name  # noqa: E402

RUN_FILES = (
    "data/labeled_train.jsonl",
    "data/unlabeled_pool.jsonl",
    "data/unlabeled_gold.jsonl",
    "data/split_stats.json",
    "pseudo/pseudo_pool.jsonl",
    "pseudo/pseudo_filtered.jsonl",
    "pseudo/pseudo_stats.json",
)
LABELED_FILES = ("train.jsonl", "dev.jsonl", "test.jsonl", "build_stats.json")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_checked(src: Path, dst: Path, overwrite: bool) -> str:
    """Copy src to dst. Returns 'copied' / 'same'; raises if dst differs and overwrite is off."""
    if not src.exists():
        raise SystemExit(f"[import] 缺少源文件：{src}")
    if dst.exists():
        if md5(src) == md5(dst):
            return "same"
        if not overwrite:
            raise SystemExit(f"[import] {dst} 已存在且内容不同；确认要覆盖时加 --overwrite")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def validate_run(paths: RunPaths, train_ids: set[str]) -> dict:
    labeled = {str(r["id"]) for r in load_jsonl(paths.labeled_train)}
    pool = [r for r in load_jsonl(paths.unlabeled_pool)]
    pool_ids = {str(r["id"]) for r in pool}
    gold_ids = {str(r["id"]) for r in load_jsonl(paths.unlabeled_gold)}
    pseudo = load_jsonl(paths.pseudo_pool)
    pseudo_ids = {str(r["id"]) for r in pseudo}
    problems = []
    if pool_ids != gold_ids:
        problems.append("无标签池与金标签的 id 不一致")
    if pool_ids != pseudo_ids:
        problems.append("无标签池与伪标签池的 id 不一致")
    if labeled & pool_ids:
        problems.append(f"有标签部分与无标签池重叠 {len(labeled & pool_ids)} 条")
    if not (labeled | pool_ids) <= train_ids:
        problems.append(f"{len((labeled | pool_ids) - train_ids)} 条不在 processed/labeled/train.jsonl 中")
    missing_fields = [f for f in ("pseudo_label", "confidence", "logic_score", "probs") if pseudo and f not in pseudo[0]]
    if missing_fields:
        problems.append(f"伪标签池缺少字段 {missing_fields}")
    if problems:
        raise SystemExit(f"[import] {paths.root.name} 校验失败：" + "；".join(problems))
    return {"labeled": len(labeled), "unlabeled_pool": len(pool_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import v2 splits and pseudo-label pools into v3.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="v3 config")
    parser.add_argument("--v2_root", type=str, default=str(ROOT.parent / "v2"))
    parser.add_argument("--v2_config", type=str, default="configs/config_sci.yaml", help="relative to --v2_root")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    v2_root = Path(args.v2_root).resolve()
    with open(v2_root / args.v2_config, encoding="utf-8") as fh:
        import yaml
        v2_cfg = yaml.safe_load(fh)

    # 1. settings that determine the pool must be identical
    diffs = diff_settings(pool_settings(v2_cfg), pool_settings(cfg))
    if diffs:
        print("[import] v2 与 v3 中决定伪标签池的设置不同，v2 的伪标签池不能复用：")
        for d in diffs:
            print(f"   - {d}   (v2 != v3)")
        raise SystemExit(1)
    print("[import] ✅ v2 与 v3 决定伪标签池的设置完全相同")

    # 2. labeled data
    v2_processed = (v2_root / v2_cfg["paths"]["processed_dir"]).resolve() / "labeled"
    v3_labeled = labeled_split_path(cfg, "train").parent
    for name in LABELED_FILES:
        status = copy_checked(v2_processed / name, v3_labeled / name, args.overwrite)
        print(f"[import] processed/labeled/{name}: {status}")
    train_ids = {str(r["id"]) for r in load_jsonl(v3_labeled / "train.jsonl")}

    # 3. runs
    v2_runs = (v2_root / v2_cfg["paths"]["runs_dir"]).resolve()
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        src_root = v2_runs / run_name(args.ratio, seed)
        dst_paths = RunPaths(run_dir(cfg, args.ratio, seed))
        if not src_root.exists():
            raise SystemExit(f"[import] v2 没有这个 run：{src_root}")
        copied = {}
        for rel in RUN_FILES:
            copied[rel] = copy_checked(src_root / rel, dst_paths.root / rel, args.overwrite)
        ext_src = src_root / "outputs" / "extractor"
        for f in sorted(ext_src.glob("*")) if ext_src.exists() else []:
            if f.is_file():
                copied[f"outputs/extractor/{f.name}"] = copy_checked(f, dst_paths.extractor_outputs / f.name, args.overwrite)
        sizes = validate_run(dst_paths, train_ids)
        marker = {
            "source_run": str(src_root),
            "source_config": str(v2_root / args.v2_config),
            "imported": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": {rel: md5(dst_paths.root / rel) for rel in copied},
            **sizes,
        }
        with open(dst_paths.import_marker, "w", encoding="utf-8") as fh:
            json.dump(marker, fh, indent=2, ensure_ascii=False)
        n_new = sum(v == "copied" for v in copied.values())
        print(f"[import] ✅ {dst_paths.root.name}: 有标签 {sizes['labeled']} 条，无标签池 {sizes['unlabeled_pool']} 条；"
              f"复制 {n_new} 个文件，{len(copied) - n_new} 个已存在且相同")


if __name__ == "__main__":
    main()
