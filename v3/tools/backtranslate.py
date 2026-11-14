"""
backtranslate.py (README_v3 14.3, 14.4)
=======================================
FixMatch / FreeMatch（方法 FIX / FREE）的强增强：把无标签池和验证集的 claim 与证据经德语、俄语各回译一次，
缓存在 processed/backtranslation/<lang>.jsonl（每行 {"src": 原文, "bt": 回译}），然后做 14.4 的两项检查：

  第 2 项  回译输出：没有空串；回译与原文的词数比在 [0.5, 2] 之间的单位占 95% 以上（否则退出码 1）。
  第 1 项  回译是否改变标签：在验证集上用 models.nli_model 零样本预测原文与三种回译方式
           （V1 claim 和证据都回译、V2 只回译证据、V3 只回译 claim），算方向翻转率
           （原文预测为 entailment / contradiction 的样本中，回译后变成相反方向的比例），
           按 V1 → V2 → V3 的顺序取第一个两种语言都 ≤ 5% 的方式；都不满足时取平均翻转率最低的。
           结果写进 processed/backtranslation/check.json；选定的方式由人填进 common/paths.py 的 SSL_SETTINGS。

单位：claim 整句；证据每段只取前 BT_MAX_WORDS 个词，按句切分后逐句翻译再拼回。范围：runs 目录中
ratio 0.1、seed 42–46 的无标签池，加上验证集；不翻译有标签训练集和测试集。已缓存的单位不重新翻译。

Usage:
    python tools/backtranslate.py --config configs/config_fc.yaml --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import bt_unit, label_to_id, load_jsonl  # noqa: E402
from common.paths import (  # noqa: E402
    BT_MAX_WORDS,
    BT_MODELS,
    SSL_SETTINGS,
    RunPaths,
    backtranslation_path,
    cfg_path,
    labeled_split_path,
    load_config,
    run_dir,
)
from models.logic_scorer import LogicScorer  # noqa: E402

MAX_EVIDENCES = 5                 # 与 ClaimEvidenceDataset、compute_nli_probs.py 相同
MAX_FLIP_RATE = 0.05              # 14.4 第 1 项
WORD_RATIO_RANGE = (0.5, 2.0)     # 14.4 第 2 项
MIN_RATIO_SHARE = 0.95
VARIANTS = {"V1": "claim 和证据都回译", "V2": "只回译证据", "V3": "只回译 claim"}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"“'(\[]?[A-Z0-9])")


def evidence_units(rec: dict) -> list[str]:
    """The evidence segments the detector can see, each cut to its first BT_MAX_WORDS words."""
    evidences = rec.get("evidence", []) or []
    if isinstance(evidences, str):
        evidences = [evidences]
    return [bt_unit(e, BT_MAX_WORDS) for e in evidences[:MAX_EVIDENCES]]


def split_sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT.split(text) if s.strip()]


def translate(texts: list[str], model_name: str, cache_dir: str, device: torch.device, batch_size: int) -> tuple[list[str], int]:
    """Translate with the model's own generation settings; longest first so memory problems show up at once."""
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()
    lengths = [len(ids) for ids in tok(texts, truncation=False)["input_ids"]]
    n_truncated = sum(n > tok.model_max_length for n in lengths)
    order = sorted(range(len(texts)), key=lambda i: -lengths[i])
    out: list[str] = [""] * len(texts)
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc)
        for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = t.strip()
        if (b // batch_size) % 50 == 0:
            print(f"    {model_name}: {min(b + batch_size, len(order))}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"    {model_name}: {len(texts)} texts in {time.time() - t0:.0f}s, {n_truncated} longer than "
          f"{tok.model_max_length} tokens (truncated)", flush=True)
    del model
    torch.cuda.empty_cache()
    return out, n_truncated


def back_translate_units(units: list[tuple[str, str]], lang: str, cache_dir: str, device: torch.device,
                         batch_size: int) -> tuple[dict[str, str], dict]:
    """units: (kind, text). Claims are translated whole; evidence sentence by sentence and rejoined."""
    pieces: dict[str, None] = {}
    plan: list[tuple[str, list[str]]] = []
    for kind, text in units:
        parts = [text] if kind == "claim" else split_sentences(text)
        plan.append((text, parts))
        for p in parts:
            pieces.setdefault(p, None)
    plist = list(pieces)
    fwd, bwd = BT_MODELS[lang]
    print(f"  [{lang}] {len(units)} units -> {len(plist)} unique sentences", flush=True)
    mid, trunc_fwd = translate(plist, fwd, cache_dir, device, batch_size)
    back, trunc_bwd = translate(mid, bwd, cache_dir, device, batch_size)
    bt_of = dict(zip(plist, back))
    result = {text: " ".join(bt_of[p] for p in parts) for text, parts in plan}
    return result, {"sentences": len(plist), "truncated_forward": trunc_fwd, "truncated_backward": trunc_bwd}


def collect_units(cfg: dict, ratio: float, seeds: list[int]) -> tuple[list[tuple[str, str]], list[dict]]:
    records = []
    for s in seeds:
        pool = RunPaths(run_dir(cfg, ratio, s)).unlabeled_pool
        if not pool.exists():
            raise SystemExit(f"unlabeled pool not found: {pool}")
        records += load_jsonl(pool)
    dev = load_jsonl(labeled_split_path(cfg, "dev"))
    seen: dict[str, str] = {}
    for rec in records + dev:
        seen.setdefault(rec["claim"], "claim")
        for e in evidence_units(rec):
            seen.setdefault(e, "evidence")
    return [(kind, text) for text, kind in seen.items()], dev


def output_check(units: list[tuple[str, str]], bt: dict[str, str], lang: str) -> dict:
    """14.4 item 2."""
    empty = [t for _, t in units if not bt.get(t, "").strip()]
    ratios = np.array([len(bt[t].split()) / max(len(t.split()), 1) for _, t in units if t in bt])
    share = float(np.mean((ratios >= WORD_RATIO_RANGE[0]) & (ratios <= WORD_RATIO_RANGE[1])))
    rnd = random.Random(0)
    print(f"  [{lang}] empty outputs: {len(empty)}; word ratio in {list(WORD_RATIO_RANGE)}: {share:.1%} "
          f"(median {np.median(ratios):.2f}); examples:")
    for kind, text in rnd.sample(units, 5):
        print(f"    ({kind}) {text[:200]}\n      -> {bt[text][:200]}")
    return {"empty": len(empty), "word_ratio_share": share, "word_ratio_median": float(np.median(ratios)),
            "ok": not empty and share >= MIN_RATIO_SHARE}


def variant_pair(rec: dict, bt: dict[str, str] | None, variant: str) -> tuple[str, str]:
    """(claim, premise) for the NLI check; premise = evidence units joined by spaces (compute_nli_probs.py)."""
    ev = evidence_units(rec)
    bt_claim = bt is not None and variant in ("V1", "V3")
    bt_ev = bt is not None and variant in ("V1", "V2")
    claim = bt[rec["claim"]] if bt_claim else rec["claim"]
    premise = " ".join(bt[e] for e in ev) if bt_ev else " ".join(ev)
    return claim, premise


def label_check(cfg: dict, dev: list[dict], bts: dict[str, dict[str, str]], device: str) -> dict:
    """14.4 item 1: does back-translation flip the NLI direction on the dev set?"""
    scorer = LogicScorer(model_name=cfg["models"]["nli_model"], device=device, cache_dir=cfg_path(cfg, "model_cache_dir"),
                         local_files_only=bool(cfg.get("use_local_models", False)))
    gold = np.array([label_to_id(r["label"]) for r in dev])

    def predict(pairs):
        return np.array([int(np.argmax(p)) for p in scorer.probs_batch(pairs, batch_size=32, desc="NLI")])

    orig = predict([variant_pair(r, None, "V1") for r in dev])
    directional = orig != 2   # 0 = SUPPORTS (entailment), 1 = REFUTES (contradiction), 2 = NEI (neutral)
    report = {"n_dev": len(dev), "n_directional": int(directional.sum()),
              "orig_macro_f1": float(f1_score(gold, orig, average="macro", labels=[0, 1, 2], zero_division=0)),
              "variants": {}}
    print(f"  original: NLI zero-shot dev macro-F1 {report['orig_macro_f1']:.4f}; "
          f"{report['n_directional']} / {len(dev)} predicted entailment or contradiction")
    for v in VARIANTS:
        report["variants"][v] = {}
        for lang, bt in bts.items():
            pred = predict([variant_pair(r, bt, v) for r in dev])
            flips = directional & (pred != 2) & (pred != orig)
            row = {"flip_rate": float(flips.sum() / max(directional.sum(), 1)), "agreement": float(np.mean(pred == orig)),
                   "macro_f1": float(f1_score(gold, pred, average="macro", labels=[0, 1, 2], zero_division=0))}
            report["variants"][v][lang] = row
            print(f"  {v}（{VARIANTS[v]}）{lang}: direction flips {row['flip_rate']:.2%}, argmax agreement "
                  f"{row['agreement']:.2%}, dev macro-F1 {row['macro_f1']:.4f}")
    chosen = next((v for v in VARIANTS if all(r["flip_rate"] <= MAX_FLIP_RATE for r in report["variants"][v].values())), None)
    rule = f"第一个两种语言方向翻转率都 ≤ {MAX_FLIP_RATE:.0%} 的方式"
    if chosen is None:
        chosen = min(VARIANTS, key=lambda v: np.mean([r["flip_rate"] for r in report["variants"][v].values()]))
        rule = f"没有方式满足 ≤ {MAX_FLIP_RATE:.0%}，取平均翻转率最低的"
    report.update({"chosen": chosen, "rule": rule})
    print(f"  chosen: {chosen}（{VARIANTS[chosen]}）— {rule}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Back-translation for FIX / FREE and the checks of README_v3 14.4.")
    p.add_argument("--config", default="configs/config_fc.yaml")
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    cache_dir = str(cfg_path(cfg, "model_cache_dir"))
    units, dev = collect_units(cfg, args.ratio, [int(s) for s in args.seeds.split(",")])
    print(f"[backtranslate] {sum(k == 'claim' for k, _ in units)} claims + {sum(k == 'evidence' for k, _ in units)} "
          f"evidence segments (first {BT_MAX_WORDS} words) from ratio {args.ratio} pools {args.seeds} + dev", flush=True)

    bts, stats = {}, {"units": len(units), "langs": {}}
    for lang in SSL_SETTINGS["bt_langs"]:
        path = backtranslation_path(cfg, lang)
        cached = {r["src"]: r["bt"] for r in load_jsonl(path)} if path.exists() else {}
        todo = [(k, t) for k, t in units if t not in cached]
        if todo:
            new, info = back_translate_units(todo, lang, cache_dir, device, args.batch_size)
            cached.update(new)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                for text, out in cached.items():
                    fh.write(json.dumps({"src": text, "bt": out}, ensure_ascii=False) + "\n")
            stats["langs"][lang] = {"translated": len(todo), **info}
        else:
            print(f"  [{lang}] all {len(units)} units cached in {path}")
            stats["langs"][lang] = {"translated": 0}
        bts[lang] = cached
        stats["langs"][lang]["output_check"] = output_check(units, cached, lang)

    stats["label_check"] = label_check(cfg, dev, bts, args.device)
    out = backtranslation_path(cfg, "check").with_suffix(".json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(f"[backtranslate] wrote {out}")
    if not all(v["output_check"]["ok"] for v in stats["langs"].values()):
        print("[backtranslate] 回译输出检查未通过（README_v3 14.4 第 2 项），停下检查")
        sys.exit(1)


if __name__ == "__main__":
    main()
