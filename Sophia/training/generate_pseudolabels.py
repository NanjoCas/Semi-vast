"""
generate_pseudolabels.py
========================
Phase 2: Generate pseudo-labels for the unlabeled pool using the trained
TextualFeatureExtractor, combined with LogicScorer and DiscourseScorer signals.

Usage:
    python training/generate_pseudolabels.py --config configs/config.yaml
    python training/generate_pseudolabels.py --config configs/config.yaml \
        --checkpoint checkpoints/extractor/best_model.pt \
        --threshold 0.5 --device cuda

Outputs (relative to project root):
    processed/pseudo_labels/pseudo_labeled_pool.jsonl    (all samples)
    processed/pseudo_labels/pseudo_labeled_filtered.jsonl (weight >= threshold)
    processed/pseudo_labels/processing_stats.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on sys.path so run_pipeline.py and models/ are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_pipeline import UnlabeledClaimDataset                   # noqa: E402
from models.extractor import TextualFeatureExtractor             # noqa: E402
from models.logic_scorer import LogicScorer                      # noqa: E402
from models.discourse_scorer import DiscourseScorer              # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ID2LABEL = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO"}
DEFAULT_WEIGHT_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# DiscourseScorer adapter
# ---------------------------------------------------------------------------

class _DiscourseScorerAdapter:
    """
    Thin wrapper around DiscourseScorer that makes its ``score_batch`` method
    return ``list[float]`` (the raw ``discourse_score`` value) instead of
    ``list[dict]``, which is what TextualFeatureExtractor.generate_pseudo_labels
    expects.
    """

    def __init__(self, scorer: DiscourseScorer) -> None:
        self._scorer = scorer

    def score_batch(self, claims: list[str]) -> list[float]:
        results = self._scorer.score_batch(claims)
        return [r["discourse_score"] for r in results]

    def score(self, claim: str) -> float:
        return self._scorer.score(claim)["discourse_score"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(cli_device: str | None) -> torch.device:
    """Return the best available device, respecting an optional CLI override."""
    if cli_device:
        return torch.device(cli_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_statistics(results: list[dict], title: str = "Statistics") -> None:
    """Print label distribution and average score metrics for a result set."""
    if not results:
        print(f"  [{title}] Empty result set — nothing to report.")
        return

    label_counts = Counter(r["pseudo_label"] for r in results)
    total = len(results)

    avg_confidence = sum(r["confidence"] for r in results) / total
    avg_logic = sum(r["logic_score"] for r in results) / total
    avg_weight = sum(r["weight"] for r in results) / total
    avg_entropy = sum(r["entropy"] for r in results) / total

    print(f"\n  [{title}]  total={total}")
    print("  Label distribution:")
    for label_id in sorted(label_counts):
        cnt = label_counts[label_id]
        label_name = ID2LABEL.get(label_id, str(label_id))
        print(f"    {label_name:20s}: {cnt:6d}  ({cnt / total * 100:.1f}%)")
    print(f"  avg_confidence  : {avg_confidence:.4f}")
    print(f"  avg_logic_score : {avg_logic:.4f}")
    print(f"  avg_weight      : {avg_weight:.4f}")
    print(f"  avg_entropy     : {avg_entropy:.4f}")


def compute_labeled_class_priors(
    labeled_train_path: Path,
    num_classes: int = 3,
    smoothing: float = 1e-3,
) -> list[float]:
    """Estimate class priors from labeled train.jsonl with additive smoothing."""
    id_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
    counts = [0.0] * num_classes

    with open(labeled_train_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            raw = rec.get("label", "NOT_ENOUGH_INFO")
            if isinstance(raw, int):
                y = min(max(int(raw), 0), num_classes - 1)
            else:
                y = id_map.get(str(raw), 2)
            counts[y] += 1.0

    counts = [c + float(smoothing) for c in counts]
    total = sum(counts)
    return [c / total for c in counts]


def save_jsonl(records: list[dict], path: Path) -> None:
    """Write a list of dicts to a .jsonl file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records to {path}")


def save_pseudolabel_plots(
    results: list[dict],
    filtered_results: list[dict],
    output_dir: Path,
) -> list[Path]:
    """Save publication-ready pseudo-label plots (PNG + PDF)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    labels = [0, 1, 2]
    label_names = [ID2LABEL[l] for l in labels]

    full_counts = [sum(1 for r in results if r.get("pseudo_label") == l) for l in labels]
    filt_counts = [sum(1 for r in filtered_results if r.get("pseudo_label") == l) for l in labels]

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width / 2 for i in x], full_counts, width=width, color="tab:blue", alpha=0.85, label="Full Pool")
    ax.bar([i + width / 2 for i in x], filt_counts, width=width, color="tab:orange", alpha=0.85, label="Filtered Pool")
    ax.set_xticks(list(x))
    ax.set_xticklabels(label_names)
    ax.set_ylabel("Count")
    ax.set_title("Pseudo-Label Distribution: Full vs Filtered")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    dist_png = output_dir / "pseudolabel_label_distribution.png"
    dist_pdf = output_dir / "pseudolabel_label_distribution.pdf"
    fig.savefig(dist_png, dpi=300, bbox_inches="tight")
    fig.savefig(dist_pdf, bbox_inches="tight")
    plt.close(fig)
    created.extend([dist_png, dist_pdf])

    full_weights = [float(r.get("weight", 0.0)) for r in results]
    filt_weights = [float(r.get("weight", 0.0)) for r in filtered_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    if full_weights:
        ax.hist(full_weights, bins=40, alpha=0.6, color="tab:blue", label="Full Pool", density=True)
    if filt_weights:
        ax.hist(filt_weights, bins=40, alpha=0.6, color="tab:orange", label="Filtered Pool", density=True)
    ax.set_title("Composite Weight Distribution")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    w_png = output_dir / "pseudolabel_weight_distribution.png"
    w_pdf = output_dir / "pseudolabel_weight_distribution.pdf"
    fig.savefig(w_png, dpi=300, bbox_inches="tight")
    fig.savefig(w_pdf, bbox_inches="tight")
    plt.close(fig)
    created.extend([w_png, w_pdf])

    return created


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Generate pseudo-labels for the unlabeled pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g. configs/config.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to the trained extractor checkpoint. "
            "Defaults to checkpoints/extractor/best_model.pt (relative to project root)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_WEIGHT_THRESHOLD,
        help=(
            "Minimum composite weight to include a sample in the filtered output. "
            f"Defaults to {DEFAULT_WEIGHT_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (e.g. 'cuda', 'cpu', 'mps').",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference. Overrides config value when provided.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Load configuration ──────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    training_cfg = config["training"]
    paths_cfg = config["paths"]
    models_cfg = config["models"]
    hp_cfg = config.get("hyperparameters", {})
    imbalance_cfg = config.get("imbalance", {})
    algorithm_cfg = config.get("algorithm", {})

    # ── Resolve paths ───────────────────────────────────────────────────────
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else PROJECT_ROOT / paths_cfg["checkpoints"] / "extractor" / "best_model.pt"
    )
    unlabeled_pool_path = PROJECT_ROOT / paths_cfg["unlabeled_pool"]
    output_dir = PROJECT_ROOT / "processed" / "pseudo_labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_output_path = output_dir / "pseudo_labeled_pool.jsonl"
    filtered_output_path = output_dir / "pseudo_labeled_filtered.jsonl"
    stats_output_path = output_dir / "processing_stats.json"

    weight_threshold: float = args.threshold

    # ── Device ─────────────────────────────────────────────────────────────
    device = resolve_device(args.device)
    print(f"[generate_pseudolabels] Using device: {device}")

    # ── Batch size ──────────────────────────────────────────────────────────
    batch_size: int = (
        args.batch_size
        if args.batch_size is not None
        else training_cfg.get("batch_size", 64)
    )
    print(f"[generate_pseudolabels] Batch size: {batch_size}")

    # ── Load extractor ──────────────────────────────────────────────────────
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Extractor checkpoint not found: {checkpoint_path}\n"
            "Run `python training/train_extractor.py` first to train the model."
        )

    model_name: str = models_cfg["deberta_base"]
    print(f"[generate_pseudolabels] Initialising TextualFeatureExtractor ({model_name})...")
    extractor = TextualFeatureExtractor(model_name=model_name, num_labels=3)
    extractor.load(str(checkpoint_path), device=device)
    extractor.to(device)
    extractor.eval()

    # ── Load auxiliary scorers ───────────────────────────────────────────────
    nli_model: str = models_cfg["nli_model"]
    print(f"[generate_pseudolabels] Loading LogicScorer ({nli_model})...")
    logic_scorer = LogicScorer(model_name=nli_model, device=device)

    print("[generate_pseudolabels] Loading DiscourseScorer...")
    discourse_scorer = _DiscourseScorerAdapter(DiscourseScorer())

    # ── Load unlabeled dataset ───────────────────────────────────────────────
    if not unlabeled_pool_path.exists():
        raise FileNotFoundError(
            f"Unlabeled pool not found: {unlabeled_pool_path}\n"
            "Run `python run_pipeline.py` first to generate the processed datasets."
        )

    max_length: int = training_cfg.get("max_length", 256)
    from transformers import AutoTokenizer
    print(f"[generate_pseudolabels] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"[generate_pseudolabels] Loading unlabeled pool from: {unlabeled_pool_path}")
    unlabeled_dataset = UnlabeledClaimDataset(
        str(unlabeled_pool_path),
        tokenizer,
        max_length=max_length,
    )
    print(f"[generate_pseudolabels] Pool size: {len(unlabeled_dataset)} samples")

    # ── Optional prior-adjusted pseudo labeling for imbalance ───────────────
    apply_prior_adjust = bool(
        imbalance_cfg.get("apply_prior_adjust_in_pseudolabels", True)
    )
    logit_adjust_tau = float(algorithm_cfg.get("logit_adjust_tau", 0.0))
    class_priors: list[float] | None = None
    if apply_prior_adjust and logit_adjust_tau > 0.0:
        labeled_train_path = PROJECT_ROOT / paths_cfg["labeled_train"]
        if not labeled_train_path.exists():
            raise FileNotFoundError(f"Labeled train data not found: {labeled_train_path}")
        class_priors = compute_labeled_class_priors(labeled_train_path)
        print(
            "[generate_pseudolabels] Prior-adjusted pseudo labeling enabled: "
            f"tau={logit_adjust_tau:.3f}, priors={class_priors}"
        )
    else:
        print("[generate_pseudolabels] Prior-adjusted pseudo labeling disabled.")

    # ── Composite weights from config ────────────────────────────────────────
    beta1: float = float(hp_cfg.get("beta1", 0.5))
    beta2: float = float(hp_cfg.get("beta2", 0.3))
    beta3: float = float(hp_cfg.get("beta3", 0.2))
    print(
        f"[generate_pseudolabels] Score weights — "
        f"beta1 (confidence)={beta1}, beta2 (logic)={beta2}, beta3 (discourse)={beta3}"
    )

    # ── Generate pseudo-labels ───────────────────────────────────────────────
    print("\n[generate_pseudolabels] Running pseudo-label generation...\n")
    results: list[dict] = extractor.generate_pseudo_labels(
        unlabeled_dataset=unlabeled_dataset,
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

    # Attach human-readable label strings for convenience
    for r in results:
        r["pseudo_label_str"] = ID2LABEL.get(r["pseudo_label"], str(r["pseudo_label"]))

    # ── Statistics ───────────────────────────────────────────────────────────
    print_statistics(results, title="Full pool")

    # ── Filter by composite weight ───────────────────────────────────────────
    filtered_results = [r for r in results if r["weight"] >= weight_threshold]
    retention_rate = len(filtered_results) / max(len(results), 1)
    print(
        f"\n[generate_pseudolabels] Filtering with weight >= {weight_threshold}: "
        f"{len(filtered_results)}/{len(results)} samples retained "
        f"({retention_rate * 100:.1f}%)"
    )
    print_statistics(filtered_results, title="Filtered pool")

    # ── Save outputs ─────────────────────────────────────────────────────────
    print("\n[generate_pseudolabels] Saving outputs...")
    save_jsonl(results, all_output_path)
    save_jsonl(filtered_results, filtered_output_path)

    # ── Build and save processing stats ──────────────────────────────────────
    label_counts_all = Counter(r["pseudo_label"] for r in results)
    label_counts_filtered = Counter(r["pseudo_label"] for r in filtered_results)

    def _label_dist(counts: Counter, total: int) -> dict:
        return {
            ID2LABEL.get(k, str(k)): {
                "count": v,
                "fraction": round(v / max(total, 1), 4),
            }
            for k, v in sorted(counts.items())
        }

    stats = {
        "checkpoint": str(checkpoint_path),
        "unlabeled_pool": str(unlabeled_pool_path),
        "weight_threshold": weight_threshold,
        "score_weights": {"beta1": beta1, "beta2": beta2, "beta3": beta3},
        "prior_adjustment": {
            "enabled": bool(class_priors is not None and logit_adjust_tau > 0.0),
            "logit_adjust_tau": logit_adjust_tau,
            "class_priors": class_priors,
        },
        "total_samples": len(results),
        "filtered_samples": len(filtered_results),
        "retention_rate": round(retention_rate, 4),
        "full_pool": {
            "label_distribution": _label_dist(label_counts_all, len(results)),
            "avg_confidence": round(sum(r["confidence"] for r in results) / max(len(results), 1), 4),
            "avg_logic_score": round(sum(r["logic_score"] for r in results) / max(len(results), 1), 4),
            "avg_weight": round(sum(r["weight"] for r in results) / max(len(results), 1), 4),
            "avg_entropy": round(sum(r["entropy"] for r in results) / max(len(results), 1), 4),
        },
        "filtered_pool": {
            "label_distribution": _label_dist(label_counts_filtered, len(filtered_results)),
            "avg_confidence": round(sum(r["confidence"] for r in filtered_results) / max(len(filtered_results), 1), 4),
            "avg_logic_score": round(sum(r["logic_score"] for r in filtered_results) / max(len(filtered_results), 1), 4),
            "avg_weight": round(sum(r["weight"] for r in filtered_results) / max(len(filtered_results), 1), 4),
            "avg_entropy": round(sum(r["entropy"] for r in filtered_results) / max(len(filtered_results), 1), 4),
        },
    }

    with open(stats_output_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(f"  Saved processing stats to {stats_output_path}")

    # Save publication-ready plots
    try:
        figure_paths = save_pseudolabel_plots(results, filtered_results, output_dir)
        if figure_paths:
            print("[generate_pseudolabels] Training plots saved:")
            for fig_path in figure_paths:
                print(f"  - {fig_path}")
    except Exception as e:
        print(f"[generate_pseudolabels] Plot generation skipped due to error: {e}")

    print("\n[generate_pseudolabels] Done.")
    print(f"  Full pool    -> {all_output_path}")
    print(f"  Filtered set -> {filtered_output_path}")
    print(f"  Stats        -> {stats_output_path}")


if __name__ == "__main__":
    main()
