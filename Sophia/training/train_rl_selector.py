"""
train_rl_selector.py
====================
Phase 3: Train the PPO Reinforced Selector to filter pseudo-labeled samples.

The selector learns a policy that assigns inclusion weights to samples drawn
from a pseudo-labeled pool. Quality is measured by the macro-F1 of a
lightweight LogisticRegression probe trained on DeBERTa [CLS] embeddings of
the labeled train split augmented with the currently selected pseudo samples,
then evaluated on the labeled dev (val) split.

Usage:
    python training/train_rl_selector.py --config configs/config.yaml

Outputs:
    processed/pseudo_labels/rl_selected.jsonl   -- selected pseudo samples
    checkpoints/rl_selector/ppo_model.pt         -- saved PPO policy weights
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Project-root imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_pipeline import ClaimEvidenceDataset          # noqa: E402
from models.rl_selector import PPOSelector              # noqa: E402
from models.extractor import TextualFeatureExtractor    # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_rl_selector")

# ---------------------------------------------------------------------------
# Label mapping (kept local to avoid circular imports at top level)
# ---------------------------------------------------------------------------
LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}


# ============================================================================
# Helper: resolve device
# ============================================================================

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Helper: load raw JSONL records
# ============================================================================

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================================
# DeBERTa [CLS] embedding extraction
# ============================================================================

def _extract_cls_embeddings(
    records: list[dict],
    extractor: TextualFeatureExtractor,
    tokenizer,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 32,
    evidence_sep: str = " [SEP] ",
    max_evidences: int = 3,
    desc: str = "Extracting CLS embeddings",
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> np.ndarray:
    """
    Extract DeBERTa [CLS] embeddings for a list of records.

    Each record may contain:
        - "claim"    (str)           : the claim text
        - "evidence" (list[str])     : optional list of evidence strings

    Returns a float32 numpy array of shape (N, hidden_size).
    """
    extractor.eval()
    extractor.to(device)

    all_embeddings: list[np.ndarray] = []
    total = len(records)

    log.info("%s (%d samples, batch_size=%d) …", desc, total, batch_size)
    t0 = time.time()

    for start in range(0, total, batch_size):
        batch_records = records[start : start + batch_size]

        texts_a: list[str] = []
        texts_b: list[str] = []

        for rec in batch_records:
            claim = rec.get("claim", "")
            evidences = rec.get("evidence", [])[:max_evidences]
            evidence_text = evidence_sep.join(evidences) if evidences else ""
            texts_a.append(claim)
            texts_b.append(evidence_text)

        encoding = tokenizer(
            texts_a,
            texts_b,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.no_grad():
            kwargs: dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(use_amp and device.type == "cuda"),
                dtype=amp_dtype,
            ):
                outputs = extractor.deberta(**kwargs)
                cls_vecs = outputs.last_hidden_state[:, 0, :]  # (B, H)

        all_embeddings.append(cls_vecs.cpu().float().numpy())

        if (start // batch_size + 1) % 10 == 0:
            pct = min(start + batch_size, total) / total * 100
            log.info("  %.1f%% (%d/%d)", pct, min(start + batch_size, total), total)

    elapsed = time.time() - t0
    log.info("  Done in %.1fs", elapsed)
    return np.concatenate(all_embeddings, axis=0)


# ============================================================================
# Build val_f1_fn
# ============================================================================

def build_val_f1_fn(
    labeled_train_records: list[dict],
    val_records: list[dict],
    extractor: TextualFeatureExtractor,
    tokenizer,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 32,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Callable[[list[dict]], float]:
    """
    Build and return a val_f1_fn callable with cached labeled-train embeddings.

    The returned function:
        1. Concatenates labeled-train CLS embeddings with embeddings of the
           provided pseudo-labeled selection.
        2. Trains a LogisticRegression probe on the combined embeddings.
        3. Evaluates on the val set and returns macro-F1.

    Labeled-train embeddings are computed once and cached in the closure to
    avoid repeated DeBERTa forward passes.
    """
    # -- Extract and cache labeled val embeddings --
    log.info("Pre-computing val embeddings …")
    val_embeddings = _extract_cls_embeddings(
        val_records,
        extractor,
        tokenizer,
        device,
        max_length=max_length,
        batch_size=batch_size,
        desc="Val embeddings",
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    val_labels = np.array(
        [LABEL2ID.get(r.get("label", "NOT_ENOUGH_INFO"), 2) for r in val_records],
        dtype=np.int64,
    )

    # -- Extract and cache labeled train embeddings --
    log.info("Pre-computing labeled-train embeddings (cached) …")
    train_embeddings = _extract_cls_embeddings(
        labeled_train_records,
        extractor,
        tokenizer,
        device,
        max_length=max_length,
        batch_size=batch_size,
        desc="Labeled train embeddings",
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    train_labels = np.array(
        [LABEL2ID.get(r.get("label", "NOT_ENOUGH_INFO"), 2) for r in labeled_train_records],
        dtype=np.int64,
    )

    log.info(
        "Embedding cache ready: train=%d, val=%d",
        len(train_labels),
        len(val_labels),
    )

    def val_f1_fn(selected_pseudo: list[dict]) -> float:
        """
        Compute macro-F1 on the val set using a LogisticRegression probe.

        Args:
            selected_pseudo: list of pseudo-labeled sample dicts.
                Each dict must have a "claim" key and a "pseudo_label" key
                (int: 0/1/2) or a "label" key (str).

        Returns:
            float: macro-F1 score in [0, 1].
        """
        if selected_pseudo:
            pseudo_embeddings = _extract_cls_embeddings(
                selected_pseudo,
                extractor,
                tokenizer,
                device,
                max_length=max_length,
                batch_size=batch_size,
                desc="  Pseudo embeddings for probe",
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            # Normalise pseudo label: accept int pseudo_label or str label
            pseudo_labels_list: list[int] = []
            for rec in selected_pseudo:
                raw = rec.get("pseudo_label")
                if isinstance(raw, int):
                    pseudo_labels_list.append(raw)
                elif isinstance(raw, str):
                    pseudo_labels_list.append(LABEL2ID.get(raw, 2))
                else:
                    str_label = rec.get("label", "NOT_ENOUGH_INFO")
                    pseudo_labels_list.append(LABEL2ID.get(str_label, 2))
            pseudo_labels = np.array(pseudo_labels_list, dtype=np.int64)

            X_train = np.concatenate([train_embeddings, pseudo_embeddings], axis=0)
            y_train = np.concatenate([train_labels, pseudo_labels], axis=0)
        else:
            X_train = train_embeddings
            y_train = train_labels

        clf = LogisticRegression(max_iter=200, n_jobs=-1)
        clf.fit(X_train, y_train)

        preds = clf.predict(val_embeddings)
        macro_f1: float = f1_score(val_labels, preds, average="macro", zero_division=0)
        return macro_f1

    return val_f1_fn


def _normalize_label_id(rec: dict) -> int:
    raw = rec.get("pseudo_label")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        return LABEL2ID.get(raw, 2)
    return LABEL2ID.get(rec.get("label", "NOT_ENOUGH_INFO"), 2)


def save_selector_plots(
    pseudo_pool: list[dict],
    selected_samples: list[dict],
    outputs_dir: Path,
) -> list[Path]:
    """Save publication-ready selection plots (PNG + PDF)."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # Figure 1: label distribution (pool vs selected)
    labels = [0, 1, 2]
    label_names = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

    pool_counts = [sum(1 for r in pseudo_pool if _normalize_label_id(r) == lid) for lid in labels]
    sel_counts = [sum(1 for r in selected_samples if _normalize_label_id(r) == lid) for lid in labels]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, pool_counts, width=width, label="Pseudo Pool", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, sel_counts, width=width, label="RL Selected", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylabel("Count")
    ax.set_title("Pseudo-Label Distribution: Pool vs RL Selected")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    dist_png = outputs_dir / "rl_selector_label_distribution.png"
    dist_pdf = outputs_dir / "rl_selector_label_distribution.pdf"
    fig.savefig(dist_png, dpi=300, bbox_inches="tight")
    fig.savefig(dist_pdf, bbox_inches="tight")
    plt.close(fig)
    created.extend([dist_png, dist_pdf])

    # Figure 2: selected sample quality histograms
    weights = [float(r.get("weight", 0.0)) for r in selected_samples]
    logic_abs = [abs(float(r.get("logic_score", 0.0))) for r in selected_samples]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    if weights:
        ax1.hist(weights, bins=30, color="tab:green", alpha=0.85, edgecolor="white")
    ax1.set_title("Selected Sample Weights")
    ax1.set_xlabel("Weight")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, linestyle="--", alpha=0.35)

    if logic_abs:
        ax2.hist(logic_abs, bins=30, color="tab:purple", alpha=0.85, edgecolor="white")
    ax2.set_title("Selected |LogicScore| Distribution")
    ax2.set_xlabel("|LogicScore|")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    qual_png = outputs_dir / "rl_selector_selected_quality.png"
    qual_pdf = outputs_dir / "rl_selector_selected_quality.pdf"
    fig.savefig(qual_png, dpi=300, bbox_inches="tight")
    fig.savefig(qual_pdf, bbox_inches="tight")
    plt.close(fig)
    created.extend([qual_png, qual_pdf])

    return created


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: Train PPO Reinforced Selector to filter pseudo-labeled samples."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--extractor-ckpt",
        type=str,
        default="checkpoints/extractor/best_model.pt",
        help="Path to trained TextualFeatureExtractor checkpoint.",
    )
    parser.add_argument(
        "--pseudo-pool",
        type=str,
        default=None,
        help="Override path to pseudo_labeled_filtered.jsonl.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cpu / cuda / mps). Auto-detected when omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override embedding extraction batch size.",
    )
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    project_root = Path(__file__).parent.parent

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    pseudo_dir = project_root / "processed" / "pseudo_labels"
    labeled_dir = project_root / "processed" / "labeled"

    pseudo_pool_path: Path
    if args.pseudo_pool:
        pseudo_pool_path = Path(args.pseudo_pool)
    else:
        pseudo_pool_path = pseudo_dir / "pseudo_labeled_filtered.jsonl"

    if not pseudo_pool_path.exists():
        # Fallback: try unfiltered pseudo labels
        fallback = pseudo_dir / "pseudo_labeled.jsonl"
        if fallback.exists():
            log.warning(
                "pseudo_labeled_filtered.jsonl not found; falling back to %s",
                fallback,
            )
            pseudo_pool_path = fallback
        else:
            log.error(
                "No pseudo-label file found. Expected: %s", pseudo_pool_path
            )
            sys.exit(1)

    train_path = labeled_dir / "train.jsonl"
    dev_path = labeled_dir / "dev.jsonl"

    for p in (train_path, dev_path):
        if not p.exists():
            log.error("Required labeled data file not found: %s", p)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device(args.device) if args.device else _resolve_device()
    log.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Hyper-parameters from config
    # ------------------------------------------------------------------
    rl_cfg = cfg.get("rl", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("models", {})

    state_dim: int = rl_cfg.get("state_dim", 4)
    action_dim: int = rl_cfg.get("action_dim", 2)
    ppo_lr: float = rl_cfg.get("ppo_lr", 3e-4)
    ppo_epochs: int = rl_cfg.get("ppo_epochs", 4)
    clip_epsilon: float = rl_cfg.get("clip_epsilon", 0.2)
    gamma: float = rl_cfg.get("gamma", 0.99)
    gae_lambda: float = rl_cfg.get("gae_lambda", 0.95)
    n_steps: int = rl_cfg.get("n_steps", 512)

    deberta_model_name: str = model_cfg.get("deberta_base", "microsoft/deberta-v3-base")
    max_length: int = train_cfg.get("max_length", 512)
    embed_batch_size: int = args.batch_size or train_cfg.get("batch_size", 32)
    use_tf32 = bool(train_cfg.get("use_tf32", True))
    use_bf16 = bool(train_cfg.get("use_bf16", False))
    use_fp16 = bool(train_cfg.get("use_fp16", False))
    use_amp = bool(use_bf16 or use_fp16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        log.info("CUDA mixed precision for embedding extraction: amp=%s dtype=%s tf32=%s", use_amp, amp_dtype, use_tf32)

    # Seed
    seed: int = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    log.info("Loading pseudo-label pool from %s …", pseudo_pool_path)
    pseudo_pool = _load_jsonl(pseudo_pool_path)
    log.info("  %d pseudo-labeled samples loaded.", len(pseudo_pool))

    log.info("Loading labeled train from %s …", train_path)
    labeled_train = _load_jsonl(train_path)
    log.info("  %d labeled train samples loaded.", len(labeled_train))

    log.info("Loading labeled dev from %s …", dev_path)
    labeled_dev = _load_jsonl(dev_path)
    log.info("  %d labeled dev samples loaded.", len(labeled_dev))

    # ------------------------------------------------------------------
    # Load tokenizer + extractor
    # ------------------------------------------------------------------
    log.info("Loading DeBERTa tokenizer: %s", deberta_model_name)
    tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)

    log.info("Initialising TextualFeatureExtractor …")
    extractor = TextualFeatureExtractor(model_name=deberta_model_name)

    extractor_ckpt = Path(args.extractor_ckpt)
    if extractor_ckpt.exists():
        log.info("Loading extractor weights from %s", extractor_ckpt)
        extractor.load(str(extractor_ckpt), device=device)
    else:
        log.warning(
            "Extractor checkpoint not found at %s — using random weights. "
            "Embeddings will not be meaningful.",
            extractor_ckpt,
        )

    extractor.to(device)
    extractor.eval()

    # Disable gradient computation for the extractor throughout this script
    for param in extractor.parameters():
        param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Build val_f1_fn
    # ------------------------------------------------------------------
    log.info("Building val_f1_fn (this will pre-compute train & val embeddings) …")
    val_f1_fn = build_val_f1_fn(
        labeled_train_records=labeled_train,
        val_records=labeled_dev,
        extractor=extractor,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=embed_batch_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    # ------------------------------------------------------------------
    # Initialise PPOSelector
    # ------------------------------------------------------------------
    log.info(
        "Initialising PPOSelector (state_dim=%d, action_dim=%d) …",
        state_dim,
        action_dim,
    )
    selector = PPOSelector(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=ppo_lr,
        ppo_epochs=ppo_epochs,
        clip_epsilon=clip_epsilon,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_steps=n_steps,
        device=device,
    )

    # ------------------------------------------------------------------
    # Run selection
    # ------------------------------------------------------------------
    log.info("Running PPO-based sample selection on %d candidates …", len(pseudo_pool))
    selected_samples: list[dict] = selector.select(pseudo_pool, val_f1_fn)
    log.info("Selection complete: %d / %d samples retained.", len(selected_samples), len(pseudo_pool))

    # ------------------------------------------------------------------
    # Compute and print selection statistics
    # ------------------------------------------------------------------
    if selected_samples:
        selection_ratio = len(selected_samples) / len(pseudo_pool)
        avg_weight = float(np.mean([s.get("weight", 0.0) for s in selected_samples]))
        avg_logic_score = float(np.mean([s.get("logic_score", 0.0) for s in selected_samples]))
    else:
        selection_ratio = 0.0
        avg_weight = 0.0
        avg_logic_score = 0.0

    print("\n" + "=" * 60)
    print("RL Selector — Selection Summary")
    print("=" * 60)
    print(f"  Pseudo pool size   : {len(pseudo_pool)}")
    print(f"  Selected samples   : {len(selected_samples)}")
    print(f"  Selection ratio    : {selection_ratio:.4f}  ({selection_ratio * 100:.1f}%)")
    print(f"  Avg weight         : {avg_weight:.4f}")
    print(f"  Avg logic_score    : {avg_logic_score:.4f}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Save selected samples
    # ------------------------------------------------------------------
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    rl_selected_path = pseudo_dir / "rl_selected.jsonl"

    log.info("Saving selected samples to %s …", rl_selected_path)
    with open(rl_selected_path, "w", encoding="utf-8") as fh:
        for rec in selected_samples:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("  Saved %d records.", len(selected_samples))

    # ------------------------------------------------------------------
    # Save PPO model checkpoint
    # ------------------------------------------------------------------
    ckpt_dir = project_root / "checkpoints" / "rl_selector"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ppo_ckpt_path = ckpt_dir / "ppo_model.pt"

    log.info("Saving PPO model to %s …", ppo_ckpt_path)
    selector.save(str(ppo_ckpt_path))
    log.info("  Saved.")

    # ------------------------------------------------------------------
    # Save publication-ready plots
    # ------------------------------------------------------------------
    outputs_dir = project_root / "outputs"
    try:
        figure_paths = save_selector_plots(pseudo_pool, selected_samples, outputs_dir)
        if figure_paths:
            log.info("Saved RL selector plots:")
            for fp in figure_paths:
                log.info("  - %s", fp)
    except Exception as e:
        log.warning("Plot generation skipped due to error: %s", e)

    print("Phase 3 complete.")
    print(f"  RL-selected samples : {rl_selected_path}")
    print(f"  PPO checkpoint      : {ppo_ckpt_path}")


if __name__ == "__main__":
    main()
