"""
汇总 TAPT 可行性测试（README_v3 13.5.1）：每个快照的验证集 MLM loss 与微调后的最佳验证 macro-F1，
按事先写定的规则选设置，写入 <run>/outputs/tapt_probe/report.md。只读验证集结果，不读测试集。

选择规则（13.5.1）：最佳验证 macro-F1 最高者；与最高值相差不到 0.005 的，取 epoch 更少的，再取学习率更低的。

Usage:
    python tools/tapt_probe_report.py --run_dir runs_fc/r0.10_s42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.paths import RunPaths  # noqa: E402

TIE = 0.005


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    paths = RunPaths(args.run_dir)

    rows = []
    for res_path in sorted(paths.tapt_probe.glob("*/epoch*/val_results.json")):
        with open(res_path, encoding="utf-8") as fh:
            res = json.load(fh)
        with open(res_path.parent / "tapt_info.json", encoding="utf-8") as fh:
            info = json.load(fh)
        s = info["settings"]
        dev_loss = next(h["dev_mlm_loss"] for h in info["history"] if h.get("epoch") == info["epoch"] and h["phase"] == "full")
        rows.append({"lr": s["lr"], "mask_prob": s["mask_prob"], "epoch": info["epoch"], "full_steps": info["full_steps"],
                     "dev_mlm_loss": dev_loss, "head_dev_mlm_loss": next(h["dev_mlm_loss"] for h in info["history"] if h["phase"] == "head"),
                     "val_macro_f1": res["best_val"]["macro_f1"], "best_step": res["best_step"]})
    if not rows:
        raise SystemExit(f"no probe results under {paths.tapt_probe}")

    a_val = None
    a_res = paths.detector_outputs("A") / "test_results.json"
    if a_res.exists():
        with open(a_res, encoding="utf-8") as fh:
            a_val = json.load(fh)["best_val"]["macro_f1"]   # 只取验证集结果

    top = max(r["val_macro_f1"] for r in rows)
    near = [r for r in rows if r["val_macro_f1"] >= top - TIE]
    chosen = min(near, key=lambda r: (r["epoch"], r["lr"], -r["val_macro_f1"]))

    rows.sort(key=lambda r: (r["mask_prob"], r["lr"], r["epoch"]))
    lines = [
        f"# TAPT 可行性测试（{paths.root.name}，README_v3 13.5.1）",
        "",
        f"A 的最佳验证 macro-F1：{a_val:.4f}" if a_val is not None else "A 的结果不存在",
        "",
        "| 学习率 | 遮盖比例 | epoch | 步数 | 验证集 MLM loss（只训头后 → 该 epoch） | 最佳验证 macro-F1 | 与 A 之差 | 最佳步 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = " ✓" if r is chosen else ""
        diff = f"{r['val_macro_f1'] - a_val:+.4f}" if a_val is not None else "—"
        lines.append(f"| {r['lr']:g} | {r['mask_prob']:g} | {r['epoch']} | {r['full_steps']} | "
                     f"{r['head_dev_mlm_loss']:.3f} → {r['dev_mlm_loss']:.3f} | {r['val_macro_f1']:.4f}{mark} | {diff} | {r['best_step']} |")
    lines += ["", f"选中（规则：最高值，相差 < {TIE} 时取 epoch 少、学习率低的）：学习率 {chosen['lr']:g}，"
                  f"遮盖比例 {chosen['mask_prob']:g}，{chosen['epoch']} 个 epoch（验证 macro-F1 {chosen['val_macro_f1']:.4f}）"]
    text = "\n".join(lines) + "\n"
    out = paths.tapt_probe / "report.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(json.dumps({"lr": chosen["lr"], "mask_prob": chosen["mask_prob"], "epochs": chosen["epoch"]}))


if __name__ == "__main__":
    main()
