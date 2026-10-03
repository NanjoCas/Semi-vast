"""
generate_pseudolabels.py (v2)
=============================
Phase 2: Generate pseudo-labels for the run's unlabeled pool.

v2 changes vs. v1:
  - The pool is claim + evidence pairs (same format as dev/test), encoded with
    ClaimEvidenceDataset, so the extractor predicts in-distribution.
  - LogicScore uses the sample's own evidence as the NLI premise (v1 used "").
  - Outputs carry `evidence` so the RL probe and the detector see the same pairs.
  - Never reads unlabeled_gold.jsonl.

Outputs (inside --run_dir):
    pseudo/pseudo_pool.jsonl       all unlabeled samples with pseudo-labels and scores
    pseudo/pseudo_filtered.jsonl   weight >= experiment.weight_threshold (input to RL and method W)
    pseudo/pseudo_stats.json

Usage:
    python training/generate_pseudolabels.py --run_dir runs/r0.10_s42 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import ClaimEvidenceDataset, ID2LABEL, load_jsonl, save_jsonl  # noqa: E402
from common.paths import RunPaths, cfg_path, load_config  # noqa: E402
from models.discourse_scorer import DiscourseScorer  # noqa: E402
from models.extractor import TextualFeatureExtractor  # noqa: E402
from models.logic_scorer import LogicScorer  # noqa: E402
from common.fingerprint import pseudolabel_prior_tau  # noqa: E402


class _DiscourseScorerAdapter:
    """DiscourseScorer.score_batch returns dicts; the extractor expects floats."""

    def __init__(self, scorer: DiscourseScorer) -> None:
        self._scorer = scorer

    def score_batch(self, claims: list[str]) -> list[float]:
        return [r["discourse_score"] for r in self._scorer.score_batch(claims)]


def resolve_device(cli_device: str | None) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_labeled_class_priors(records: list[dict], smoothing: float = 1e-3) -> list[float]:
    id_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
    counts = [0.0, 0.0, 0.0]
    for rec in records:
        counts[id_map.get(str(rec.get("label")), 2)] += 1.0
    counts = [c + smoothing for c in counts]
    total = sum(counts)
    return [c / total for c in counts]


def summarize(results: list[dict]) -> dict:
    n = max(len(results), 1)
    return {
        "count": len(results),
        "label_distribution": {
            ID2LABEL[k]: v for k, v in sorted(Counter(r["pseudo_label"] for r in results).items())
        },
        "avg_confidence": round(sum(r["confidence"] for r in results) / n, 4),
        "avg_logic_score": round(sum(r["logic_score"] for r in results) / n, 4),
        "avg_abs_logic_score": round(sum(abs(r["logic_score"]) for r in results) / n, 4),
        "avg_weight": round(sum(r["weight"] for r in results) / n, 4),
        "avg_entropy": round(sum(r["entropy"] for r in results) / n, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 (v2): pseudo-label the unlabeled claim+evidence pool.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override experiment.weight_threshold.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    training_cfg = cfg["training"]
    models_cfg = cfg["models"]
    hp_cfg = cfg.get("hyperparameters", {})
    imbalance_cfg = cfg.get("imbalance", {})
    exp_cfg = cfg.get("experiment", {})

    paths = RunPaths(args.run_dir)
    if not paths.extractor_ckpt.exists():
        raise FileNotFoundError(f"Extractor checkpoint not found: {paths.extractor_ckpt}")
    if not paths.unlabeled_pool.exists():
        raise FileNotFoundError(f"Unlabeled pool not found: {paths.unlabeled_pool}")

    device = resolve_device(args.device)
    batch_size = args.batch_size or int(training_cfg.get("batch_size", 16)) * 2
    weight_threshold = float(args.threshold if args.threshold is not None else exp_cfg.get("weight_threshold", 0.4))
    max_length = int(training_cfg.get("max_length", 384))

    model_name = models_cfg["deberta_base"]
    model_cache_dir = cfg_path(cfg, "model_cache_dir")
    local_only = bool(cfg.get("use_local_models", False))
    cache_kwargs = {"cache_dir": str(model_cache_dir)}
    if local_only:
        cache_kwargs["local_files_only"] = True

    print(f"[generate_pseudolabels] device={device} batch_size={batch_size} threshold={weight_threshold}")

    extractor = TextualFeatureExtractor(
        model_name=model_name,
        num_labels=3,
        cache_dir=model_cache_dir,
        local_files_only=local_only,
    )
    extractor.load(str(paths.extractor_ckpt), device=device)
    extractor.to(device).eval()

    logic_scorer = LogicScorer(
        model_name=models_cfg["nli_model"],
        device=device,
        cache_dir=model_cache_dir,
        local_files_only=local_only,
    )
    discourse_scorer = _DiscourseScorerAdapter(DiscourseScorer())

    tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_kwargs)
    pool_dataset = ClaimEvidenceDataset(
        str(paths.unlabeled_pool),
        tokenizer,
        max_length=max_length,
        require_label=False,
    )
    print(f"[generate_pseudolabels] Pool size: {len(pool_dataset)} (claim + evidence pairs)")

    class_priors = None
    # v3: the prior correction of pseudo labels has its own key (imbalance.pseudolabel_prior_tau);
    # in v2 it shared algorithm.logit_adjust_tau with the detector loss.
    prior_tau = pseudolabel_prior_tau(cfg)
    logit_adjust_tau = prior_tau or 0.0
    if prior_tau is not None:
        # Priors come from the run's labeled part only.
        class_priors = compute_labeled_class_priors(load_jsonl(paths.labeled_train))
        print(f"[generate_pseudolabels] Prior adjustment tau={logit_adjust_tau:.3f} priors={class_priors}")

    beta1 = float(hp_cfg.get("beta1", 0.5))
    beta2 = float(hp_cfg.get("beta2", 0.3))
    beta3 = float(hp_cfg.get("beta3", 0.2))

    results = extractor.generate_pseudo_labels(
        unlabeled_dataset=pool_dataset,
        logic_scorer=logic_scorer,
        discourse_scorer=discourse_scorer,
        batch_size=batch_size,
        device=device,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        class_priors=class_priors,
        logit_adjust_tau=logit_adjust_tau,
    )
    source_by_id = {str(r["id"]): r.get("source", "unknown") for r in pool_dataset.records}
    for r in results:
        r["source"] = source_by_id.get(str(r["id"]), "unknown")
        r["pseudo_label_str"] = ID2LABEL.get(r["pseudo_label"], str(r["pseudo_label"]))

    filtered = [r for r in results if r["weight"] >= weight_threshold]

    save_jsonl(results, paths.pseudo_pool)
    save_jsonl(filtered, paths.pseudo_filtered)

    stats = {
        "run_dir": str(paths.root),
        "weight_threshold": weight_threshold,
        "score_weights": {"beta1": beta1, "beta2": beta2, "beta3": beta3},
        "prior_adjustment": {"enabled": class_priors is not None, "tau": logit_adjust_tau, "priors": class_priors},
        "retention_rate": round(len(filtered) / max(len(results), 1), 4),
        "full_pool": summarize(results),
        "filtered_pool": summarize(filtered),
    }
    with open(paths.pseudo_stats, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if stats["retention_rate"] < 0.1:
        # With real evidence |LogicScore| is no longer saturated at ~1 (v1 used an
        # empty premise), so composite weights are lower than in v1.
        print(
            f"[generate_pseudolabels] WARNING: only {stats['retention_rate'] * 100:.1f}% of the pool has "
            f"weight >= {weight_threshold}. Check the weight distribution in pseudo_stats.json and "
            "consider lowering experiment.weight_threshold."
        )
    print(f"[generate_pseudolabels] pool -> {paths.pseudo_pool}")
    print(f"[generate_pseudolabels] filtered ({len(filtered)}) -> {paths.pseudo_filtered}")


if __name__ == "__main__":
    main()
