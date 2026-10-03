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

4. 数据来源可配置（data.sources，默认 [climate_fever, pubhealth]）
   [climate_fever, climatecheck, scifact]：去掉 PUBHEALTH，改用证据为科学摘要的
   ClimateCheck 与 SciFact（见 README 第 8 节）。两者的标签针对 (claim, 摘要) 对，
   同一 claim 的多个摘要用 "group" 标记，划分时始终在同一侧；ClimateCheck 中改写自
   Climate-FEVER 的 claim 会被去重，保证同一 claim 不跨 split。

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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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


def _norm_claim(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


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
    # DISPUTED = evidence both supports and refutes. v2 mapped it to REFUTES; without a
    # "mixed" class in ClimateCheck/SciFact it can also be dropped or treated as NEI.
    disputed = str(data_cfg.get("cf_disputed", "refutes")).lower()
    if disputed not in ("refutes", "nei", "drop"):
        raise ValueError(f"data.cf_disputed must be refutes | nei | drop, got {disputed!r}")

    records = []
    skipped = dropped_disputed = 0
    for item in items:
        claim = _clean_text(item.get("claim", ""))
        raw_label = item.get("claim_label", "")
        if raw_label == "DISPUTED" and disputed == "drop":
            dropped_disputed += 1
            continue
        if raw_label == "DISPUTED" and disputed == "nei":
            label = "NOT_ENOUGH_INFO"
        else:
            label = CLIMATE_FEVER_LABEL_MAP.get(raw_label, None)
        candidates = [
            _truncate_words(_clean_text(ev.get("evidence", "")), max_words)
            for ev in item.get("evidences", [])
            if _clean_text(ev.get("evidence", ""))
        ]
        if not claim or label is None or not candidates:
            skipped += 1
            continue
        rec_id = f"cf_{item.get('claim_id', _stable_id('cf', claim))}"
        records.append({
            "id": rec_id,
            "group": rec_id,
            "claim": claim,
            "evidence": ranker.top_k(claim, candidates, k),
            "label": label,
            "source": "climate_fever",
        })

    dup_dropped = conflict_dropped = 0
    if bool(data_cfg.get("dedup_claims", False)):
        # The raw file repeats some claims under different claim_ids (e.g. cf_100 / cf_1507);
        # split by normalised text so a claim never sits in two splits. Conflicting labels
        # for the same text are dropped altogether.
        by_text: dict[str, list[dict]] = {}
        for r in records:
            by_text.setdefault(_norm_claim(r["claim"]), []).append(r)
        kept = []
        for members in by_text.values():
            if len({m["label"] for m in members}) > 1:
                conflict_dropped += len(members)
            else:
                kept.append(members[0])
                dup_dropped += len(members) - 1
        records = kept

    labels = [r["label"] for r in records]
    train, temp, _, temp_labels = train_test_split(
        records, labels, test_size=0.30, stratify=labels, random_state=seed
    )
    dev, test = train_test_split(temp, test_size=0.5, stratify=temp_labels, random_state=seed)
    stats = {
        "total": len(records),
        "skipped": skipped,
        "disputed": disputed,
        "dropped_disputed": dropped_disputed,
        "dropped_duplicate_claims": dup_dropped,
        "dropped_conflicting_claims": conflict_dropped,
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
# ClimateCheck / SciFact: (claim, scientific abstract) pairs
# ─────────────────────────────────────────────────────────────

CLIMATECHECK_LABEL_MAP = {
    "Supports": "SUPPORTS",
    "Refutes": "REFUTES",
    "Not Enough Information": "NOT_ENOUGH_INFO",
}
SCIFACT_LABEL_MAP = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"}


# OpenAlex abstracts in ClimateCheck often start with an "Abstract" heading, sometimes glued
# to the text ("AbstractThere are ...") or repeated ("Abstract Abstract Emissions ...").
# The heading is only stripped before whitespace, punctuation or an uppercase letter, so
# "Abstracts of ..." is left alone.
_ABSTRACT_HEADING = re.compile(r"^(?:(?:Abstract|ABSTRACT|abstract)(?=[\sA-Z:.\-—]|$)[\s:.\-—]*)+")


def _clean_abstract(text) -> str:
    return _ABSTRACT_HEADING.sub("", _clean_text(text))


def _split_by_group(records: list[dict], dev_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Carve dev out of records by claim group: all abstracts of a claim stay on one side."""
    groups = sorted({r["group"] for r in records})
    random.Random(seed).shuffle(groups)
    dev_groups = set(groups[: int(round(len(groups) * dev_ratio))])
    return [r for r in records if r["group"] not in dev_groups], [r for r in records if r["group"] in dev_groups]


def _pair_stats(splits: dict) -> dict:
    return {
        split: {
            "pairs": len(rs),
            "claims": len({r["group"] for r in rs}),
            "label_dist": dict(Counter(r["label"] for r in rs)),
        }
        for split, rs in splits.items()
    }


def build_climatecheck(data_dir: Path, data_cfg: dict) -> tuple[dict, dict]:
    """
    Official test (gold labels published June 2026) stays test; dev is carved from the
    official train by claim. Labels belong to (claim, abstract) pairs, so one claim can
    carry different labels for different abstracts. Evidence is the whole abstract.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"ClimateCheck 目录未找到：{data_dir}")
    seed = int(data_cfg.get("split_seed", 42))
    dev_ratio = float(data_cfg.get("dev_ratio", 0.15))

    def load(split: str) -> tuple[list[dict], int]:
        files = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"ClimateCheck {split}-*.parquet 未找到：{data_dir}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        out, skipped = [], 0
        for r in df.itertuples(index=False):
            claim, abstract = _clean_text(r.claim), _clean_abstract(r.abstract)
            label = CLIMATECHECK_LABEL_MAP.get(r.annotation)
            if not claim or not abstract or label is None:
                skipped += 1
                continue
            out.append({
                "id": f"cc_{r.claim_id}_{r.abstract_id}",
                "group": f"cc_{r.claim_id}",
                "claim": claim,
                "evidence": [abstract],
                "label": label,
                "source": "climatecheck",
            })
        return out, skipped

    train_all, skipped_train = load("train")
    test, skipped_test = load("test")
    train, dev = _split_by_group(train_all, dev_ratio, seed)
    splits = {"train": train, "dev": dev, "test": test}
    return splits, {"skipped": skipped_train + skipped_test, "per_split": _pair_stats(splits)}


def build_scifact(data_dir: Path, data_cfg: dict) -> tuple[dict, dict]:
    """
    One record per (claim, cited abstract): SUPPORT/CONTRADICT when the abstract carries
    rationales, otherwise NOT_ENOUGH_INFO. claims_test.jsonl has no labels, so the
    official dev becomes our test and dev is carved from the official train by claim.
    Evidence is the whole abstract (label-blind; rationale sentences are not used).
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"SciFact 目录未找到：{data_dir}")
    seed = int(data_cfg.get("split_seed", 42))
    dev_ratio = float(data_cfg.get("dev_ratio", 0.15))
    corpus = {d["doc_id"]: d for d in map(json.loads, open(data_dir / "corpus.jsonl", encoding="utf-8"))}
    missing_docs = 0

    def load(split: str) -> list[dict]:
        nonlocal missing_docs
        out = []
        with open(data_dir / f"claims_{split}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                c = json.loads(line)
                if "evidence" not in c:
                    raise ValueError(f"SciFact claims_{split}.jsonl 没有标签")
                for doc in c.get("cited_doc_ids", []):
                    if doc not in corpus:
                        missing_docs += 1
                        continue
                    ev = c["evidence"].get(str(doc), [])
                    out.append({
                        "id": f"sf_{c['id']}_{doc}",
                        "group": f"sf_{c['id']}",
                        "claim": _clean_text(c["claim"]),
                        "evidence": [_clean_text(" ".join(corpus[doc]["abstract"]))],
                        "label": SCIFACT_LABEL_MAP[ev[0]["label"]] if ev else "NOT_ENOUGH_INFO",
                        "source": "scifact",
                    })
        return out

    train, dev = _split_by_group(load("train"), dev_ratio, seed)
    splits = {"train": train, "dev": dev, "test": load("dev")}
    return splits, {"missing_docs": missing_docs, "per_split": _pair_stats(splits)}


def drop_cross_duplicates(cf_splits: dict, cc_splits: dict, threshold: float) -> dict:
    """
    ClimateCheck re-uses Climate-FEVER claims rephrased as tweets (TF-IDF cosine 0.5-0.6 for
    real rephrasings). Keep each claim family inside one split: a match with a ClimateCheck
    test claim removes the Climate-FEVER claim (the official test stays intact); a match
    across different train/dev/test splits otherwise removes the ClimateCheck claim group.
    """
    cf_recs = [(split, r) for split, rs in cf_splits.items() for r in rs]
    cc_groups: dict[str, tuple[str, str]] = {}
    for split, rs in cc_splits.items():
        for r in rs:
            cc_groups.setdefault(r["group"], (split, r["claim"]))
    if threshold <= 0 or not cf_recs or not cc_groups:
        return {"threshold": threshold, "matches": 0, "dropped_climate_fever": 0, "dropped_climatecheck_claims": 0}

    keys = list(cc_groups)
    cf_claims = [r["claim"] for _, r in cf_recs]
    cc_claims = [cc_groups[g][1] for g in keys]
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True).fit(cf_claims + cc_claims)
    sim = (vec.transform(cc_claims) @ vec.transform(cf_claims).T).toarray()

    drop_cf, drop_cc = set(), set()
    rows, cols = np.where(sim >= threshold)
    for i, j in zip(rows, cols):
        cc_split = cc_groups[keys[i]][0]
        cf_split, cf_rec = cf_recs[j]
        if cc_split == "test":
            drop_cf.add(cf_rec["id"])
        elif cf_split != cc_split:
            drop_cc.add(keys[i])
    for split in cf_splits:
        cf_splits[split] = [r for r in cf_splits[split] if r["id"] not in drop_cf]
    for split in cc_splits:
        cc_splits[split] = [r for r in cc_splits[split] if r["group"] not in drop_cc]
    return {
        "threshold": threshold,
        "matches": int(len(rows)),
        "dropped_climate_fever": len(drop_cf),
        "dropped_climatecheck_claims": len(drop_cc),
    }


def drop_cross_split_claims(built: dict) -> dict:
    """
    The same claim text can sit under different ids across the official splits (e.g. SciFact
    870/871). Keep it only in the highest-priority split (test > dev > train), so evaluation
    claims never appear in training and the official test sets stay intact.
    """
    rank = {"train": 0, "dev": 1, "test": 2}
    best: dict[str, int] = {}
    for splits in built.values():
        for split, rs in splits.items():
            for r in rs:
                key = _norm_claim(r["claim"])
                best[key] = max(best.get(key, -1), rank[split])
    dropped: Counter = Counter()
    for splits in built.values():
        for split in list(splits):
            kept = [r for r in splits[split] if best[_norm_claim(r["claim"])] == rank[split]]
            dropped[split] += len(splits[split]) - len(kept)
            splits[split] = kept
    return dict(dropped)


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

    sources = list(data_cfg.get("sources", ["climate_fever", "pubhealth"]))
    unknown = sorted(set(sources) - {"climate_fever", "pubhealth", "climatecheck", "scifact"})
    if unknown:
        raise SystemExit(f"Unknown data.sources: {unknown}")
    print(f"  数据来源：{sources}")

    built: dict[str, dict] = {}
    out_stats: dict = {"sources": sources, "merged": {}}
    mode = str(data_cfg.get("pubhealth_evidence", "main_text"))
    max_words = int(data_cfg.get("max_evidence_words", 120))

    cf_items = load_climate_fever_raw(cfg_path(cfg, "raw_climate_fever")) if "climate_fever" in sources else []
    ph_frames = load_pubhealth_raw(cfg_path(cfg, "raw_pubhealth_dir"), seed) if "pubhealth" in sources else {}

    if cf_items or ph_frames:
        # Fit one TF-IDF space over all claims and candidate sentences (no labels used).
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

    if "climate_fever" in sources:
        print("\n[Climate-FEVER]（证据按相似度选取，不看标签）...")
        built["climate_fever"], out_stats["climate_fever"] = build_climate_fever(cf_items, ranker, data_cfg)
        print(f"  {out_stats['climate_fever']['split_sizes']}  标签：{out_stats['climate_fever']['label_dist']}"
              f"  DISPUTED={out_stats['climate_fever']['disputed']}（丢弃 {out_stats['climate_fever']['dropped_disputed']}）")

    if "pubhealth" in sources:
        print(f"\n[PUBHEALTH]（证据来源：{mode}）...")
        built["pubhealth"], out_stats["pubhealth"] = build_pubhealth(ph_frames, ranker, data_cfg)
        for split, st in out_stats["pubhealth"]["per_split"].items():
            print(f"  [{split}] 保留 {st['kept']}  丢弃 {st['dropped']}")

    if "climatecheck" in sources:
        print("\n[ClimateCheck]（证据为整篇摘要；dev 按 claim 从官方 train 切出）...")
        built["climatecheck"], out_stats["climatecheck"] = build_climatecheck(cfg_path(cfg, "raw_climatecheck_dir"), data_cfg)
        for split, st in out_stats["climatecheck"]["per_split"].items():
            print(f"  [{split}] {st['pairs']} 对 / {st['claims']} 个 claim  {st['label_dist']}")

    if "scifact" in sources:
        print("\n[SciFact]（证据为整篇摘要；官方 dev 作为 test）...")
        built["scifact"], out_stats["scifact"] = build_scifact(cfg_path(cfg, "raw_scifact_dir"), data_cfg)
        for split, st in out_stats["scifact"]["per_split"].items():
            print(f"  [{split}] {st['pairs']} 对 / {st['claims']} 个 claim  {st['label_dist']}")

    if "climate_fever" in built and "climatecheck" in built:
        out_stats["cross_dedup"] = drop_cross_duplicates(
            built["climate_fever"], built["climatecheck"], float(data_cfg.get("cross_dedup_threshold", 0.5))
        )
        print(f"\n  ClimateCheck ↔ Climate-FEVER 去重：{out_stats['cross_dedup']}")

    if bool(data_cfg.get("dedup_claims", False)):
        out_stats["cross_split_dedup"] = drop_cross_split_claims(built)
        print(f"  跨 split 相同 claim 去重（保留 test > dev > train）：丢弃 {out_stats['cross_split_dedup']}")

    for split in ("train", "dev", "test"):
        combined = dedup([r for src in sources for r in built[src][split]])
        for r in combined:
            r.setdefault("group", r["id"])
        random.shuffle(combined)
        path = labeled_split_path(cfg, split)
        save_jsonl(combined, path)
        out_stats["merged"][split] = {
            "total": len(combined),
            "claims": len({r["group"] for r in combined}),
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
