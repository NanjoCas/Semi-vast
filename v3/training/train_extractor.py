"""
train_extractor.py (v2)
=======================
Phase 1: Supervised pre-training of TextualFeatureExtractor on labeled data.

v2 changes vs. v1:
  - Trains on <run_dir>/data/labeled_train.jsonl (the labeled part of the SSL split).
  - --run_dir / --seed CLI arguments; checkpoints and outputs go inside the run dir.
  - warmup is a ratio of total steps (training.warmup_ratio). With 10% labels the
    whole run is ~100-200 optimizer steps, so v1's fixed 100 warmup steps would
    cover almost all of training.
  - The random word-dropping augmentation is off by default
    (training.extractor_text_augment). It decoded the claim/evidence pair into a
    single string and re-encoded it, destroying the sentence-pair structure.
  - final_model.pt is no longer written (nothing downstream reads it).

Trains a DeBERTa-v3-base model for 3-way claim verification
(SUPPORTS / REFUTES / NOT_ENOUGH_INFO) using the ClaimEvidenceDataset.

Usage:
    python training/train_extractor.py --config configs/config.yaml \
        --run_dir runs/r0.10_s42 --seed 42 --device cuda
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import transformers
# 数据增强函数
def augment_text(text: str) -> str:
    """简单文本增强：随机删除词或同义词替换（这里用随机删除作为示例）"""
    words = text.split()
    if len(words) <= 3:
        return text
    # 随机删除 10% 的词
    keep_prob = 0.9
    augmented = [word for word in words if random.random() < keep_prob]
    return ' '.join(augmented) if augmented else text
# ---------------------------------------------------------------------------
# v2 root on sys.path so common/ and models/ are importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import ClaimEvidenceDataset                 # noqa: E402
from common.paths import RunPaths, cfg_path, labeled_split_path, load_config  # noqa: E402
from models.extractor import TextualFeatureExtractor               # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def resolve_device(cli_device: str | None) -> torch.device:
    """Return the best available device, respecting an optional CLI override."""
    if cli_device:
        device = torch.device(cli_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(
                "[train_extractor] WARNING: requested device cuda is unavailable; "
                "falling back to cpu."
            )
            return torch.device("cpu")
        if device.type == "mps" and not torch.backends.mps.is_available():
            print(
                "[train_extractor] WARNING: requested device mps is unavailable; "
                "falling back to cpu."
            )
            return torch.device("cpu")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(
    dataset: ClaimEvidenceDataset,
    num_classes: int = 3,
    device: torch.device | None = None,
    power: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute configurable class weights from the label distribution of a
    labeled dataset.

    By default this uses inverse frequency weighting:
        w[c] = (total / (num_classes * count[c])) ** power

    Args:
        dataset:     A ClaimEvidenceDataset whose records contain "label" keys.
        num_classes: Number of distinct label classes. Defaults to 3.
        device:      Target tensor device.
        power:       Weight scaling power. 1.0 = inverse frequency, 0.0 = equal.
        normalize:   If True, scale the returned weights to have mean 1.0.

    Returns:
        torch.Tensor of shape (num_classes,) with dtype float32.
    """
    label_str_to_id = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
    label_ids = [
        label_str_to_id.get(rec.get("label", "NOT_ENOUGH_INFO"), 2)
        for rec in dataset.records
    ]
    counts = Counter(label_ids)
    total = len(label_ids)

    weights = []
    for cls_idx in range(num_classes):
        cnt = counts.get(cls_idx, 1)
        weight = (total / (num_classes * cnt)) ** float(power)
        weights.append(weight)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if normalize:
        mean_weight = float(weight_tensor.mean().clamp(min=1e-6))
        weight_tensor = weight_tensor / mean_weight

    if device is not None:
        weight_tensor = weight_tensor.to(device)
    return weight_tensor


def extract_label_ids(
    dataset: ClaimEvidenceDataset,
    num_classes: int = 3,
) -> list[int]:
    """
    Convert dataset labels into integer ids in [0, num_classes).
    Unknown labels are mapped to NOT_ENOUGH_INFO (2).
    """
    label_str_to_id = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
    label_ids = [
        label_str_to_id.get(rec.get("label", "NOT_ENOUGH_INFO"), 2)
        for rec in dataset.records
    ]
    return [min(max(int(x), 0), num_classes - 1) for x in label_ids]


def build_balanced_sampler(
    label_ids: list[int],
    num_classes: int = 3,
    power: float = 1.0,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to mitigate class imbalance.

    Per-sample weight:
        w_i = (1 / count[y_i]) ** power
    """
    counts = Counter(label_ids)
    sample_weights: list[float] = []
    for y in label_ids:
        cls_count = max(int(counts.get(int(y), 1)), 1)
        sample_weights.append((1.0 / cls_count) ** float(power))
    weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(label_ids),
        replacement=True,
    )


def evaluate(
    model: TextualFeatureExtractor,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Run one pass over a DataLoader and return validation metrics.

    Returns:
        dict with keys: loss (float), accuracy (float), macro_f1 (float),
        per_class_precision (list[float]), per_class_recall (list[float]),
        per_class_f1 (list[float]), auc_ovr (float|None), confusion_matrix
        (list[list[int]]).
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[list[float]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(use_amp and device.type == "cuda"),
                dtype=amp_dtype,
            ):
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    per_class_precision = precision_arr.tolist()
    per_class_recall = recall_arr.tolist()
    per_class_f1 = f1_arr.tolist()
    support = support_arr.tolist()
    confusion = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "support": support,
        "confusion_matrix": confusion,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Supervised pre-training of TextualFeatureExtractor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file (e.g. configs/config.yaml).",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory produced by data_prep/make_ssl_split.py (e.g. runs/r0.10_s42).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override (defaults to training.seed in config).",
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
        help="Optional batch size override for training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=(
            "Optional early stopping patience in epochs. "
            "If provided, training stops after this many epochs with no validation macro-F1 improvement."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def save_training_plots(all_metrics: list[dict], outputs_dir: Path) -> list[Path]:
    """
    Save publication-ready training plots to outputs_dir.

    Returns:
        list[Path]: paths of generated figure files.
    """
    if not all_metrics:
        return []

    outputs_dir.mkdir(parents=True, exist_ok=True)
    epochs = [m["epoch"] for m in all_metrics]
    train_loss = [m["train_loss"] for m in all_metrics]
    val_loss = [m["val_loss"] for m in all_metrics]
    val_acc = [m["val_accuracy"] for m in all_metrics]
    val_macro_f1 = [m["val_macro_f1"] for m in all_metrics]
    lr = [m["lr"] for m in all_metrics]

    created: list[Path] = []

    # Figure 1: Main curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.plot(epochs, train_loss, marker="o", linewidth=2, label="Train Loss")
    ax1.plot(epochs, val_loss, marker="s", linewidth=2, label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.plot(epochs, val_acc, marker="o", linewidth=2, color="tab:green")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax3.plot(epochs, val_macro_f1, marker="o", linewidth=2, color="tab:red")
    ax3.set_title("Validation Macro-F1")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Macro-F1")
    ax3.grid(True, linestyle="--", alpha=0.4)

    ax4.plot(epochs, lr, marker="o", linewidth=2, color="tab:purple")
    ax4.set_title("Learning Rate Schedule")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Learning Rate")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    curve_png = outputs_dir / "extractor_training_curves.png"
    curve_pdf = outputs_dir / "extractor_training_curves.pdf"
    fig.savefig(curve_png, dpi=300, bbox_inches="tight")
    fig.savefig(curve_pdf, bbox_inches="tight")
    plt.close(fig)
    created.extend([curve_png, curve_pdf])

    # Figure 2: Per-class F1 (SUPPORTS / REFUTES / NOT_ENOUGH_INFO)
    fig, ax = plt.subplots(figsize=(10, 6))
    class_names = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for class_idx, class_name in enumerate(class_names):
        class_f1 = [m["val_per_class_f1"][class_idx] for m in all_metrics]
        ax.plot(
            epochs,
            class_f1,
            marker="o",
            linewidth=2,
            color=colors[class_idx],
            label=class_name,
        )

    ax.set_title("Validation Per-Class F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    class_png = outputs_dir / "extractor_val_per_class_f1.png"
    class_pdf = outputs_dir / "extractor_val_per_class_f1.pdf"
    fig.savefig(class_png, dpi=300, bbox_inches="tight")
    fig.savefig(class_pdf, bbox_inches="tight")
    plt.close(fig)
    created.extend([class_png, class_pdf])

    # Figure 3: Confusion matrix for the best validation epoch
    best_epoch = max(all_metrics, key=lambda m: m.get("val_macro_f1", -1.0))
    cm = best_epoch.get("val_confusion_matrix")
    if cm is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        class_names = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
        ax.set(
            xticks=range(len(class_names)),
            yticks=range(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title=f"Validation Confusion Matrix (best epoch {best_epoch.get('epoch')})",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = max(max(row) for row in cm) if cm else 0
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(
                    j,
                    i,
                    f"{cm[i][j]}",
                    ha="center",
                    va="center",
                    color="white" if cm[i][j] > thresh / 2 else "black",
                )

        fig.tight_layout()
        cm_png = outputs_dir / "extractor_val_confusion_matrix.png"
        cm_pdf = outputs_dir / "extractor_val_confusion_matrix.pdf"
        fig.savefig(cm_png, dpi=300, bbox_inches="tight")
        fig.savefig(cm_pdf, bbox_inches="tight")
        plt.close(fig)
        created.extend([cm_png, cm_pdf])

    return created


def main() -> None:
    args = parse_args()

    # ── Load configuration ──────────────────────────────────────────────────
    config = load_config(args.config)

    training_cfg = config["training"]
    models_cfg = config["models"]
    imbalance_cfg = config.get("imbalance", {})

    run_paths = RunPaths(args.run_dir)
    outputs_dir = run_paths.extractor_outputs
    checkpoints_dir = run_paths.extractor_ckpt.parent
    outputs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ── Seeds ──────────────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else training_cfg.get("seed", 42)
    set_seeds(seed)

    # ── Device ─────────────────────────────────────────────────────────────
    device = resolve_device(args.device)
    print(f"[train_extractor] Using device: {device}")
    use_tf32 = bool(training_cfg.get("use_tf32", True))
    use_bf16 = bool(training_cfg.get("use_bf16", False))
    use_fp16 = bool(training_cfg.get("use_fp16", False))
    use_amp = bool(use_bf16 or use_fp16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        print(f"[train_extractor] CUDA mixed precision: amp={use_amp}, dtype={amp_dtype}, tf32={use_tf32}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    model_name: str = models_cfg["deberta_base"]
    model_cache_dir = cfg_path(config, "model_cache_dir")
    use_local_models = bool(config.get("use_local_models", False))
    cache_kwargs = {"cache_dir": str(model_cache_dir)}
    if use_local_models:
        cache_kwargs["local_files_only"] = True

    print(f"[train_extractor] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_kwargs)

    # ── Datasets ───────────────────────────────────────────────────────────
    max_length: int = training_cfg.get("max_length", 512)
    batch_size: int = training_cfg.get("batch_size", 16)
    if args.batch_size is not None:
        batch_size = args.batch_size

    train_path = run_paths.labeled_train
    dev_path = labeled_split_path(config, "dev")

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            "Run data_prep/build_labeled.py and data_prep/make_ssl_split.py first."
        )
    if not dev_path.exists():
        raise FileNotFoundError(
            f"Dev data not found: {dev_path}\n"
            "Run data_prep/build_labeled.py and data_prep/make_ssl_split.py first."
        )

    print(f"[train_extractor] Loading train dataset from: {train_path}")
    train_dataset = ClaimEvidenceDataset(
        str(train_path), tokenizer, max_length=max_length
    )

    print(f"[train_extractor] Loading dev dataset from: {dev_path}")
    dev_dataset = ClaimEvidenceDataset(
        str(dev_path), tokenizer, max_length=max_length
    )

    train_label_ids = extract_label_ids(train_dataset, num_classes=3)
    train_counts = Counter(train_label_ids)
    print(
        "[train_extractor] Train label distribution: "
        f"SUPPORTS={train_counts.get(0, 0)}, "
        f"REFUTES={train_counts.get(1, 0)}, "
        f"NOT_ENOUGH_INFO={train_counts.get(2, 0)}"
    )

    # ── DataLoaders ─────────────────────────────────────────────────────────
    num_workers = int(training_cfg.get("num_workers", min(4, os.cpu_count() or 1)))
    use_balanced_extractor_sampler = bool(
        imbalance_cfg.get("use_balanced_extractor_sampler", False)
    )
    extractor_sampler_power = float(imbalance_cfg.get("extractor_sampler_power", 1.0))
    train_sampler = None
    train_shuffle = True
    if use_balanced_extractor_sampler:
        train_sampler = build_balanced_sampler(
            train_label_ids,
            num_classes=3,
            power=extractor_sampler_power,
        )
        train_shuffle = False
        print(
            "[train_extractor] Balanced sampler enabled for extractor training "
            f"(power={extractor_sampler_power:.2f})."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Class weights ───────────────────────────────────────────────────────
    class_weight_power = float(imbalance_cfg.get("class_weight_power", 1.0))
    normalize_class_weights = bool(imbalance_cfg.get("normalize_class_weights", True))
    print("[train_extractor] Computing class weights from training set...")
    class_weights = compute_class_weights(
        train_dataset,
        num_classes=3,
        device=device,
        power=class_weight_power,
        normalize=normalize_class_weights,
    )
    print(
        f"[train_extractor] Class weights: {class_weights.tolist()} "
        f"(power={class_weight_power:.2f}, normalize={normalize_class_weights})"
    )

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"[train_extractor] Initialising TextualFeatureExtractor ({model_name})...")
    freeze_layers = training_cfg.get("freeze_layers", 6)
    init_from_nli = training_cfg.get("init_from_nli", False)
    nli_model_name = models_cfg.get("nli_model", "cross-encoder/nli-deberta-v3-large")
    model = TextualFeatureExtractor(
        model_name=model_name,
        num_labels=3,
        freeze_layers=freeze_layers,
        init_from_nli=init_from_nli,
        nli_model_name=nli_model_name,
        cache_dir=model_cache_dir,
        local_files_only=use_local_models,
    )

    # Enable gradient checkpointing for small batch sizes to save memory
    print("[train_extractor] Enabling gradient checkpointing to save memory.")
    model.deberta.gradient_checkpointing_enable()

    model.to(device)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda" and use_fp16))

    # ── Loss, optimiser, scheduler ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 2e-5)),
        weight_decay=0.05,  # 增加 weight_decay 从 0.01 到 0.05 以增强正则化
    )

    # Extractor-only overrides (default: the shared values). At a 10% label ratio the labeled
    # set is a few hundred pairs, and the shared gradient_accumulation=4 leaves the extractor
    # with only ~56 optimizer steps over 8 epochs.
    max_epochs: int = int(training_cfg.get("extractor_max_epochs", training_cfg.get("max_epochs", 10)))
    gradient_accumulation: int = int(
        training_cfg.get("extractor_gradient_accumulation", training_cfg.get("gradient_accumulation", 4))
    )
    use_text_augment = bool(training_cfg.get("extractor_text_augment", False))
    early_stopping_patience: int | None = training_cfg.get("early_stopping_patience", 5)  # 设置默认 patience 为 5 以防止过拟合
    max_grad_norm: float = 1.0

    if args.patience is not None:
        early_stopping_patience = args.patience

    total_update_steps = math.ceil(len(train_loader) / gradient_accumulation) * max_epochs
    if "warmup_ratio" in training_cfg:
        warmup_steps = int(round(float(training_cfg["warmup_ratio"]) * total_update_steps))
    else:
        warmup_steps = int(training_cfg.get("warmup_steps", 200))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    # ── Training loop ───────────────────────────────────────────────────────
    best_macro_f1: float = -1.0
    epochs_since_improvement: int = 0
    best_checkpoint_path = checkpoints_dir / "best_model.pt"
    all_metrics: list[dict] = []

    patience_desc = (
        f"early_stopping_patience={early_stopping_patience}" if early_stopping_patience is not None else "early_stopping disabled"
    )
    print(
        f"\n[train_extractor] Starting training — "
        f"epochs={max_epochs}, batch_size={batch_size}, "
        f"grad_accum={gradient_accumulation}, warmup_steps={warmup_steps}, {patience_desc}\n"
    )

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        optimizer.zero_grad()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{max_epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"].to(device)

            # 数据增强：随机对 50% 的批次应用增强
            if use_text_augment and random.random() < 0.5 and step > 1:  # 跳过第一个批次
                # 这里简化，只在 CPU 上增强文本，然后重新编码
                # 注意：这会增加计算开销
                augmented_texts = []
                for i in range(len(batch["input_ids"])):
                    # 解码文本
                    text = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
                    augmented_text = augment_text(text)
                    augmented_texts.append(augmented_text)
                # 重新编码
                augmented_batch = tokenizer(augmented_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
                input_ids = augmented_batch["input_ids"].to(device)
                attention_mask = augmented_batch["attention_mask"].to(device)
                if "token_type_ids" in augmented_batch:
                    token_type_ids = augmented_batch["token_type_ids"].to(device)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(use_amp and device.type == "cuda"),
                dtype=amp_dtype,
            ):
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * gradient_accumulation

            # Optimiser step after accumulating enough gradients
            if step % gradient_accumulation == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = running_loss / step
            progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # ── Epoch validation ───────────────────────────────────────────────
        val_metrics = evaluate(
            model,
            dev_loader,
            criterion,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        epoch_time = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class_precision": val_metrics["per_class_precision"],
            "val_per_class_recall": val_metrics["per_class_recall"],
            "val_per_class_f1": val_metrics["per_class_f1"],
            "val_confusion_matrix": val_metrics["confusion_matrix"],
            "val_support": val_metrics["support"],
            "epoch_time_s": round(epoch_time, 2),
            "lr": scheduler.get_last_lr()[0],
        }
        all_metrics.append(epoch_record)

        print(
            f"  [Epoch {epoch:02d}] "
            f"train_loss={epoch_record['train_loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
            f"({epoch_time:.1f}s)"
        )

        print("  [Epoch metrics] ")
        print(
            f"    Precision  : SUPPORTS={val_metrics['per_class_precision'][0]:.4f}, "
            f"REFUTES={val_metrics['per_class_precision'][1]:.4f}, "
            f"NEI={val_metrics['per_class_precision'][2]:.4f}"
        )
        print(
            f"    Recall     : SUPPORTS={val_metrics['per_class_recall'][0]:.4f}, "
            f"REFUTES={val_metrics['per_class_recall'][1]:.4f}, "
            f"NEI={val_metrics['per_class_recall'][2]:.4f}"
        )
        print(
            f"    F1         : SUPPORTS={val_metrics['per_class_f1'][0]:.4f}, "
            f"REFUTES={val_metrics['per_class_f1'][1]:.4f}, "
            f"NEI={val_metrics['per_class_f1'][2]:.4f}"
        )
        print("    Confusion matrix (rows=true, cols=pred):")
        cm = val_metrics["confusion_matrix"]
        print("      SUPP   REF   NEI")
        for i, row in enumerate(cm):
            label_name = ["SUPP", "REF", "NEI"][i]
            print(
                f"      {label_name} "
                f"{row[0]:>5} {row[1]:>5} {row[2]:>5}"
            )

        # ── Save best checkpoint ───────────────────────────────────────────
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            epochs_since_improvement = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(
                f"  [Epoch {epoch:02d}] New best macro-F1={best_macro_f1:.4f} "
                f"— checkpoint saved to {best_checkpoint_path}"
            )
        else:
            epochs_since_improvement += 1
            print(
                f"  [Epoch {epoch:02d}] No improvement. "
                f"epochs_since_improvement={epochs_since_improvement}"
            )
            if early_stopping_patience is not None and epochs_since_improvement >= early_stopping_patience:
                print(
                    f"  [train_extractor] Early stopping triggered after {epochs_since_improvement} epochs without improvement."
                )
                break

    print(f"\n[train_extractor] Best model (macro-F1={best_macro_f1:.4f}) saved to {best_checkpoint_path}")

    # ── Save training metrics ────────────────────────────────────────────────
    metrics_path = outputs_dir / "extractor_train_metrics.json"
    summary = {
        "best_val_macro_f1": best_macro_f1,
        "total_epochs": max_epochs,
        "model_name": model_name,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "warmup_steps": warmup_steps,
        "text_augment": use_text_augment,
        "train_path": str(train_path),
        "seed": seed,
        "device": str(device),
        "best_checkpoint": str(best_checkpoint_path),
        "epochs": all_metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[train_extractor] Training metrics saved to {metrics_path}")

    # Save publication-ready visual plots
    try:
        figure_paths = save_training_plots(all_metrics, outputs_dir)
        if figure_paths:
            print("[train_extractor] Training plots saved:")
            for fig_path in figure_paths:
                print(f"  - {fig_path}")
    except Exception as e:
        print(f"[train_extractor] Plot generation skipped due to error: {e}")

    print("\n[train_extractor] Done.")


if __name__ == "__main__":
    main()
