"""
build_labeled.py
================
重新构建有标签数据（Climate-FEVER + PUBHEALTH），修复 v1 中的三处问题：

1. Climate-FEVER 证据选择泄漏
   v1 的 _select_evidence_for_claim 优先挑 evidence_label == claim_label 的证据，
   即用金标签选证据（train/dev/test 都如此）。
   v2：不看任何标签，用 TF-IDF 余弦相似度从 5 条候选证据中选 top-k。

2. PUBHEALTH 证据泄漏
   v1 把 explanation（核查记者为结论写的解释）当证据，结论几乎写在输入里。
   v2：默认从 main_text（原文正文）中按 TF-IDF 相似度选 top-k 句。
   可在 config 中设 data.pubhealth_evidence: explanation 复现旧做法做对照。

3. dev/test 中 Climate-FEVER 被重复
   v1 的 merge_labeled_datasets 对 train/dev/test 都做了 cf_weight=2 过采样，
   测试集中每条 Climate-FEVER 样本被计算两次。
   v2：这里输出的三个 split 都不含重复；过采样只在 make_ssl_split.py 中
   作用于"有标签训练部分"。

输出：
    processed/labeled/train.jsonl   （唯一样本，供 make_ssl_split.py 切分）
    processed/labeled/dev.jsonl
    processed/labeled/test.jsonl
    processed/labeled/build_stats.json

Usage:
    python data_prep/build_labeled.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))

from common.data_utils import save_jsonl  # noqa: E402
from common.paths import cfg_path, labeled_split_path, load_config  # noqa: E402

# ─────────────────────────────────────────────────────────────
# Label maps / filters (same as v1 process_labeled.py)
# ─────────────────────────────────────────────────────────────

CLIMATE_FEVER_LABEL_MAP = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "DISPUTED": "REFUTES",
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
}

PUBHEALTH_LABEL_MAP = {
    "true": "SUPPORTS",
    "false": "REFUTES",
    "unproven": "NOT_ENOUGH_INFO",
    "mixture": "NOT_ENOUGH_INFO",
}

COVID_PATTERN = re.compile(
    r"\b(covid|coronavirus|sars[\-\s]?cov[\-\s]?2?|pandemic|"
    r"lockdown|quarantine|mask mandate|pcr test|contact tracing|"
    r"mRNA vaccine|pfizer|moderna|astrazeneca|johnson.*johnson)\b",
    re.IGNORECASE,
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _clean_text(text) -> str:
    text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def _stable_id(prefix: str, text: str) -> str:
    # v1 used Python's hash() as a fallback id, which changes between processes.
    return f"{prefix}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:10]}"


def _split_sentences(text: str, min_words: int = 4) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    return [s for s in sentences if len(s.split()) >= min_words]


# ─────────────────────────────────────────────────────────────
# Label-blind evidence ranking
# ─────────────────────────────────────────────────────────────

class EvidenceRanker:
    """TF-IDF cosine ranking of candidate sentences against a claim (no labels involved)."""

    def __init__(self, corpus: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.vectorizer.fit(corpus if corpus else ["empty"])

    def top_k(self, claim: str, candidates: list[str], k: int) -> list[str]:
        if len(candidates) <= k:
            return list(candidates)
        mat = self.vectorizer.transform([claim] + candidates)
        scores = (mat[1:] @ mat[0].T).toarray().ravel()
        # Stable sort: ties keep original candidate order.
        order = np.argsort(-scores, kind="stable")[:k]
        return [candidates[i] for i in order]


# ─────────────────────────────────────────────────────────────
# Climate-FEVER
# ─────────────────────────────────────────────────────────────

def load_climate_fever_raw(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Climate-FEVER 文件未找到：{path}")
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_climate_fever(items: list[dict], ranker: EvidenceRanker, data_cfg: dict) -> tuple[dict, dict]:
    k = int(data_cfg.get("max_evidences", 3))
    max_words = int(data_cfg.get("max_evidence_words", 120))
    seed = int(data_cfg.get("split_seed", 42))

    records = []
    skipped = 0
    for item in items:
        claim = _clean_text(item.get("claim", ""))
        label = CLIMATE_FEVER_LABEL_MAP.get(item.get("claim_label", ""), None)
        candidates = [
            _truncate_words(_clean_text(ev.get("evidence", "")), max_words)
            for ev in item.get("evidences", [])
            if _clean_text(ev.get("evidence", ""))
        ]
        if not claim or label is None or not candidates:
            skipped += 1
            continue
        records.append({
            "id": f"cf_{item.get('claim_id', _stable_id('cf', claim))}",
            "claim": claim,
            "evidence": ranker.top_k(claim, candidates, k),
            "label": label,
            "source": "climate_fever",
        })

    labels = [r["label"] for r in records]
    train, temp, _, temp_labels = train_test_split(
        records, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    dev, test = train_test_split(temp, test_size=0.5, stratify=temp_labels, random_state=seed)
    stats = {
        "total": len(records),
        "skipped": skipped,
        "split_sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "label_dist": dict(Counter(labels)),
    }
    return {"train": train, "dev": dev, "test": test}, stats


# ─────────────────────────────────────────────────────────────
# PUBHEALTH
# ─────────────────────────────────────────────────────────────

def load_pubhealth_raw(data_dir: Path, seed: int) -> dict[str, pd.DataFrame]:
    """Load PUBHEALTH splits; carve dev from train when dev.tsv is missing (same as v1)."""
    if not data_dir.exists():
        raise FileNotFoundError(f"PUBHEALTH 目录未找到：{data_dir}")

    def read(split: str) -> pd.DataFrame | None:
        p = data_dir / f"{split}.tsv"
        return pd.read_csv(p, sep="\t", dtype=str) if p.exists() else None

    frames = {s: read(s) for s in ("train", "dev", "test")}
    if frames["train"] is None or frames["test"] is None:
        raise FileNotFoundError(f"PUBHEALTH 需要 train.tsv 和 test.tsv：{data_dir}")
    if frames["dev"] is None:
        print("  [PUBHEALTH] dev.tsv 未找到，从 train.tsv 切分 15% 作为验证集")
        full = frames["train"]
        full = full[full["label"].isin(PUBHEALTH_LABEL_MAP.keys())]
        frames["train"], frames["dev"] = train_test_split(
            full, test_size=0.15, stratify=full["label"].tolist(), random_state=seed
        )
    return frames


def pubhealth_candidates(row, mode: str, max_words: int) -> list[str]:
    if mode == "explanation":
        text = _clean_text(row.get("explanation", ""))
        return [_truncate_words(text, 512)] if text else []
    text = _clean_text(row.get("main_text", ""))
    return [_truncate_words(s, max_words) for s in _split_sentences(text)]


def build_pubhealth(frames: dict[str, pd.DataFrame], ranker: EvidenceRanker, data_cfg: dict) -> tuple[dict, dict]:
    mode = str(data_cfg.get("pubhealth_evidence", "main_text"))
    k = int(data_cfg.get("max_evidences", 3))
    max_words = int(data_cfg.get("max_evidence_words", 120))

    if mode == "main_text" and "main_text" not in frames["train"].columns:
        raise KeyError(
            "PUBHEALTH TSV 中没有 main_text 列。请确认数据版本，"
            "或在 config 中设 data.pubhealth_evidence: explanation（会保留泄漏）。"
        )

    out: dict[str, list[dict]] = {}
    stats: dict = {"evidence_mode": mode, "per_split": {}}
    for split, df in frames.items():
        counts = Counter()
        df = df[df["label"].isin(PUBHEALTH_LABEL_MAP.keys())]
        records = []
        for _, row in df.iterrows():
            claim = _clean_text(row.get("claim", ""))
            if not claim:
                counts["empty_claim"] += 1
                continue
            if COVID_PATTERN.search(claim) or COVID_PATTERN.search(_clean_text(row.get("subjects", ""))):
                counts["covid"] += 1
                continue
            candidates = pubhealth_candidates(row, mode, max_words)
            if not candidates:
                counts["no_evidence"] += 1
                continue
            claim_id = _clean_text(row.get("claim_id", ""))
            records.append({
                "id": f"ph_{claim_id}" if claim_id else _stable_id("ph", claim),
                "claim": claim,
                "evidence": candidates if mode == "explanation" else ranker.top_k(claim, candidates, k),
                "label": PUBHEALTH_LABEL_MAP[row["label"]],
                "source": "pubhealth",
            })
        out[split] = records
        stats["per_split"][split] = {
            "kept": len(records),
            "dropped": dict(counts),
            "label_dist": dict(Counter(r["label"] for r in records)),
        }
    return out, stats


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def dedup(records: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in records:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-free labeled splits for v2.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    seed = int(data_cfg.get("split_seed", 42))
    random.seed(seed)

    print("=" * 60)
    print("v2 有标签数据构建（修复证据泄漏）")
    print("=" * 60)

    cf_items = load_climate_fever_raw(cfg_path(cfg, "raw_climate_fever"))
    ph_frames = load_pubhealth_raw(cfg_path(cfg, "raw_pubhealth_dir"), seed)

    # Fit one TF-IDF space over all claims and candidate sentences (no labels used).
    mode = str(data_cfg.get("pubhealth_evidence", "main_text"))
    max_words = int(data_cfg.get("max_evidence_words", 120))
    corpus: list[str] = []
    for item in cf_items:
        corpus.append(_clean_text(item.get("claim", "")))
        corpus.extend(_clean_text(ev.get("evidence", "")) for ev in item.get("evidences", []))
    for df in ph_frames.values():
        for _, row in df.iterrows():
            corpus.append(_clean_text(row.get("claim", "")))
            if mode == "main_text":
                corpus.extend(pubhealth_candidates(row, mode, max_words))
    corpus = [c for c in corpus if c]
    print(f"  TF-IDF 语料：{len(corpus)} 句")
    ranker = EvidenceRanker(corpus)

    print("\n[1/2] Climate-FEVER（证据按相似度选取，不看标签）...")
    cf_splits, cf_stats = build_climate_fever(cf_items, ranker, data_cfg)
    print(f"  {cf_stats['split_sizes']}  标签：{cf_stats['label_dist']}")

    print(f"\n[2/2] PUBHEALTH（证据来源：{mode}）...")
    ph_splits, ph_stats = build_pubhealth(ph_frames, ranker, data_cfg)
    for split, s in ph_stats["per_split"].items():
        print(f"  [{split}] 保留 {s['kept']}  丢弃 {s['dropped']}")

    out_stats = {"climate_fever": cf_stats, "pubhealth": ph_stats, "merged": {}}
    for split in ("train", "dev", "test"):
        combined = dedup(cf_splits[split] + ph_splits[split])
        random.shuffle(combined)
        path = labeled_split_path(cfg, split)
        save_jsonl(combined, path)
        out_stats["merged"][split] = {
            "total": len(combined),
            "by_source": dict(Counter(r["source"] for r in combined)),
            "label_dist": dict(Counter(r["label"] for r in combined)),
        }
        print(f"\n  [{split}] {len(combined)} 条 → {path}")
        print(f"          {out_stats['merged'][split]['by_source']}  {out_stats['merged'][split]['label_dist']}")

    stats_path = labeled_split_path(cfg, "train").parent / "build_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(out_stats, fh, indent=2, ensure_ascii=False)
    print(f"\n✓ 统计信息：{stats_path}")


if __name__ == "__main__":
    main()
