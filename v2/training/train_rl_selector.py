"""
train_rl_selector.py (v2)
=========================
Phase 3: Train the PPO Reinforced Selector to filter pseudo-labeled samples.

Quality is measured by the macro-F1 of a LogisticRegression probe trained on
extractor [CLS] embeddings of the run's labeled part plus the selected pseudo
samples, evaluated on the labeled dev split.

v2 changes vs. v1:
  - Pseudo samples now carry evidence, so their probe embeddings are computed
    on the same claim+evidence input as train/dev (v1 embedded them claim-only).
  - Real [CLS] embeddings are passed to the selector for the diversity feature.
  - New selection procedure (fixed baseline, multi-episode PPO updates, final
    deterministic pass); see models/rl_selector.py.
  - Probe uses lbfgs (faster than saga on dense 1024-d features).
  - Paths come from --run_dir; training history and plots are saved per run.

Usage:
    python training/train_rl_selector.py --run_dir runs/r0.10_s42 --device cuda

Outputs (inside --run_dir):
    pseudo/set_C_rl.jsonl               selected pseudo samples (method C)
    outputs/rl_selector/selection_info.json
    outputs/rl_selector/*.png|pdf
    checkpoints/rl_selector/ppo_model.pt
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
from tqdm import tqdm

# ---------------------------------------------------------------------------
# v2 root imports
# ---------------------------------------------------------------------------
V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))

from common.data_utils import load_jsonl, save_jsonl            # noqa: E402
from common.paths import RunPaths, cfg_path, labeled_split_path, load_config  # noqa: E402
from models.rl_selector import PPOSelector                      # noqa: E402
from models.extractor import TextualFeatureExtractor            # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_rl_selector")

# Keep terminal output focused: suppress chatty third-party logs.
for noisy_name in ("httpx", "urllib3", "transformers", "huggingface_hub"):
    logging.getLogger(noisy_name).setLevel(logging.WARNING)

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

    for start in tqdm(
        range(0, total, batch_size),
        desc=desc,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
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


    elapsed = time.time() - t0
    log.info("  Done in %.1fs", elapsed)
    return np.concatenate(all_embeddings, axis=0)


# ============================================================================
# Build val_f1_fn
# ============================================================================

def build_val_f1_fn(
    labeled_train_records: list[dict],
    val_records: list[dict],
    pseudo_pool_records: list[dict],
    extractor: TextualFeatureExtractor,
    tokenizer,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 32,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Callable[[list[dict]], float], dict[str, np.ndarray]]:
    """
    Build a val_f1_fn callable with cached embeddings.

    Returns (val_f1_fn, pseudo_emb_by_key); the second item maps record keys
    to [CLS] embeddings and is reused for the selector's diversity feature.

    Caches are built once for:
      1) labeled train records
      2) labeled val records
      3) the entire pseudo-labeled pool

    This avoids repeated DeBERTa forward passes during PPO episodes.
    """

    def _record_key(rec: dict) -> str:
        rec_id = rec.get("id")
        if rec_id is not None:
            return f"id::{rec_id}"
        return f"claim::{rec.get('claim', '')}"

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

    # -- Extract and cache pseudo-pool embeddings --
    log.info("Pre-computing pseudo-pool embeddings (cached) …")
    unique_pseudo_records: list[dict] = []
    seen_keys: set[str] = set()
    for rec in pseudo_pool_records:
        key = _record_key(rec)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_pseudo_records.append(rec)

    pseudo_emb_by_key: dict[str, np.ndarray] = {}
    if unique_pseudo_records:
        pseudo_pool_embeddings = _extract_cls_embeddings(
            unique_pseudo_records,
            extractor,
            tokenizer,
            device,
            max_length=max_length,
            batch_size=batch_size,
            desc="Pseudo pool embeddings",
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        for rec, emb in zip(unique_pseudo_records, pseudo_pool_embeddings):
            pseudo_emb_by_key[_record_key(rec)] = emb

    log.info(
        "Embedding cache ready: train=%d, val=%d, pseudo=%d",
        len(train_labels),
        len(val_labels),
        len(pseudo_emb_by_key),
    )

    def val_f1_fn(selected_pseudo: list[dict]) -> float:
        """
        Compute macro-F1 on the val set using a LogisticRegression probe.
        """
        if selected_pseudo:
            # Fill cache misses only once, then use ordered lookup.
            missing_map: dict[str, dict] = {}
            for rec in selected_pseudo:
                key = _record_key(rec)
                if key not in pseudo_emb_by_key:
                    missing_map[key] = rec

            if missing_map:
                missing_records = list(missing_map.values())
                log.warning(
                    "Pseudo embedding cache miss: %d records. Computing on-the-fly.",
                    len(missing_records),
                )
                missing_embeddings = _extract_cls_embeddings(
                    missing_records,
                    extractor,
                    tokenizer,
                    device,
                    max_length=max_length,
                    batch_size=batch_size,
                    desc="  Missing pseudo embeddings",
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                for rec, emb in zip(missing_records, missing_embeddings):
                    pseudo_emb_by_key[_record_key(rec)] = emb

            ordered_embs = [pseudo_emb_by_key[_record_key(rec)] for rec in selected_pseudo]
            pseudo_embeddings = np.stack(ordered_embs, axis=0)

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

        # Standardize features for better convergence
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(val_embeddings)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        macro_f1: float = f1_score(val_labels, preds, average="macro", zero_division=0)
        return macro_f1

    return val_f1_fn, pseudo_emb_by_key


def _record_key(rec: dict) -> str:
    rec_id = rec.get("id")
    return f"id::{rec_id}" if rec_id is not None else f"claim::{rec.get('claim', '')}"


def save_history_plot(info: dict, outputs_dir: Path) -> list[Path]:
    """Reward / delta-F1 / keep-ratio per PPO iteration."""
    history = info.get("history", [])
    if not history:
        return []
    its = [h["iteration"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
        axes,
        ("reward", "delta_f1", "keep_ratio"),
        ("Episode reward", "Probe ΔF1 vs labeled-only", "Keep ratio"),
    ):
        ax.plot(its, [h[key] for h in history], marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("PPO iteration")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[1].axhline(0.0, color="grey", linewidth=1)
    fig.tight_layout()
    png = outputs_dir / "rl_selector_training_history.png"
    pdf = outputs_dir / "rl_selector_training_history.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


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
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 (v2): train the PPO Reinforced Selector.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding extraction batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = RunPaths(args.run_dir)

    rl_cfg = cfg.get("rl", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("models", {})
    hp_cfg = cfg.get("hyperparameters", {})

    for p in (paths.pseudo_filtered, paths.labeled_train, paths.extractor_ckpt):
        if not p.exists():
            log.error("Required file not found: %s", p)
            sys.exit(1)
    dev_path = labeled_split_path(cfg, "dev")

    device = torch.device(args.device) if args.device else _resolve_device()
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_bf16 = bool(train_cfg.get("use_bf16", False))
    use_fp16 = bool(train_cfg.get("use_fp16", False))
    use_amp = bool(use_bf16 or use_fp16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if device.type == "cuda":
        use_tf32 = bool(train_cfg.get("use_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32

    pseudo_pool = load_jsonl(paths.pseudo_filtered)
    labeled_train = load_jsonl(paths.labeled_train)
    labeled_dev = load_jsonl(dev_path)
    log.info("pool=%d labeled_train=%d dev=%d", len(pseudo_pool), len(labeled_train), len(labeled_dev))
    if not pseudo_pool:
        log.error("Filtered pseudo pool is empty: %s", paths.pseudo_filtered)
        sys.exit(1)

    model_name = model_cfg.get("deberta_base", "microsoft/deberta-v3-large")
    model_cache_dir = cfg_path(cfg, "model_cache_dir")
    local_only = bool(cfg.get("use_local_models", False))
    cache_kwargs = {"cache_dir": str(model_cache_dir)}
    if local_only:
        cache_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_kwargs)

    extractor = TextualFeatureExtractor(
        model_name=model_name, cache_dir=model_cache_dir, local_files_only=local_only
    )
    extractor.load(str(paths.extractor_ckpt), device=device)
    extractor.to(device).eval()
    for param in extractor.parameters():
        param.requires_grad_(False)

    val_f1_fn, pseudo_emb_by_key = build_val_f1_fn(
        labeled_train_records=labeled_train,
        val_records=labeled_dev,
        pseudo_pool_records=pseudo_pool,
        extractor=extractor,
        tokenizer=tokenizer,
        device=device,
        max_length=int(train_cfg.get("max_length", 384)),
        batch_size=args.batch_size or int(train_cfg.get("batch_size", 16)) * 2,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    embeddings = np.stack([pseudo_emb_by_key[_record_key(r)] for r in pseudo_pool], axis=0)

    selector = PPOSelector(
        state_dim=int(rl_cfg.get("state_dim", 4)),
        action_dim=int(rl_cfg.get("action_dim", 2)),
        lr=float(rl_cfg.get("ppo_lr", 3e-4)),
        ppo_epochs=int(rl_cfg.get("ppo_epochs", 4)),
        clip_epsilon=float(rl_cfg.get("clip_epsilon", 0.2)),
        gamma=float(rl_cfg.get("gamma", 1.0)),
        gae_lambda=float(rl_cfg.get("gae_lambda", 1.0)),
        alpha=float(hp_cfg.get("alpha", 0.7)),
        beta=float(hp_cfg.get("beta", 0.3)),
        seed=seed,
    )

    selected, info = selector.select(
        pseudo_pool,
        val_f1_fn,
        embeddings=embeddings,
        n_iterations=int(rl_cfg.get("n_iterations", 30)),
        episodes_per_iter=int(rl_cfg.get("episodes_per_iter", 4)),
        episode_size=int(rl_cfg.get("episode_size", 512)),
        min_keep_ratio=float(rl_cfg.get("min_keep_ratio", 0.05)),
    )
    if info.get("warning"):
        log.warning(info["warning"])

    save_jsonl(selected, paths.rl_selected)
    paths.rl_ckpt.parent.mkdir(parents=True, exist_ok=True)
    selector.save(str(paths.rl_ckpt))

    paths.rl_outputs.mkdir(parents=True, exist_ok=True)
    info["selected_label_distribution"] = {
        str(k): int(v) for k, v in zip(*np.unique([_normalize_label_id(r) for r in selected], return_counts=True))
    } if selected else {}
    with open(paths.rl_outputs / "selection_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)

    try:
        save_selector_plots(pseudo_pool, selected, paths.rl_outputs)
        save_history_plot(info, paths.rl_outputs)
    except Exception as e:  # plotting must never fail the pipeline
        log.warning("Plot generation skipped due to error: %s", e)

    print("\n" + "=" * 60)
    print("RL Selector — Selection Summary")
    print("=" * 60)
    print(f"  Pool (filtered)    : {info['pool_size']}")
    print(f"  Selected           : {info['selected']} ({info['selection_ratio'] * 100:.1f}%)")
    print(f"  Probe F1 baseline  : {info['baseline_f1']:.4f}")
    print(f"  Probe F1 selection : {info['final_selection_f1']:.4f} (Δ {info['final_delta_f1']:+.4f})")
    print(f"  keep prob mean/std : {info['keep_prob_mean']:.3f} / {info['keep_prob_std']:.3f}")
    print("=" * 60)
    print(f"  -> {paths.rl_selected}")


if __name__ == "__main__":
    main()
