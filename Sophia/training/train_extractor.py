"""
train_extractor.py
==================
Phase 1: Supervised pre-training of TextualFeatureExtractor on labeled data.

Trains a DeBERTa-v3-base model for 3-way claim verification
(SUPPORTS / REFUTES / NOT_ENOUGH_INFO) using the ClaimEvidenceDataset.

Usage:
    python training/train_extractor.py --config configs/config.yaml
    python training/train_extractor.py --config configs/config.yaml \
        --output_dir outputs/run_01 --device cuda
"""

import argparse
import json
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

# ---------------------------------------------------------------------------
# Project root on sys.path so run_pipeline.py and models/ are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_pipeline import ClaimEvidenceDataset                      # noqa: E402
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
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the label distribution of
    a labeled dataset.

    For each class c:
        weight[c] = total_samples / (num_classes * count[c])

    Args:
        dataset:     A ClaimEvidenceDataset whose records contain "label" keys.
        num_classes: Number of distinct label classes. Defaults to 3.
        device:      Target tensor device.

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
        cnt = counts.get(cls_idx, 1)          # avoid division by zero
        weights.append(total / (num_classes * cnt))

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
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
        per_class_f1 (list[float]).
    """
    from sklearn.metrics import f1_score, accuracy_score

    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

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

            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, zero_division=0
    ).tolist()

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
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
        required=True,
        help="Path to the YAML configuration file (e.g. configs/config.yaml).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override for the outputs directory defined in config.",
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

    return created


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
    imbalance_cfg = config.get("imbalance", {})

    # Resolve output directory (CLI arg takes priority over config)
    outputs_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / paths_cfg["outputs"]
    checkpoints_dir = PROJECT_ROOT / paths_cfg["checkpoints"] / "extractor"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ── Seeds ──────────────────────────────────────────────────────────────
    seed = training_cfg.get("seed", 42)
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
    print(f"[train_extractor] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Datasets ───────────────────────────────────────────────────────────
    max_length: int = training_cfg.get("max_length", 512)
    batch_size: int = training_cfg.get("batch_size", 16)
    if args.batch_size is not None:
        batch_size = args.batch_size

    train_path = PROJECT_ROOT / paths_cfg["labeled_train"]
    dev_path = PROJECT_ROOT / paths_cfg["labeled_dev"]

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            "Run `python run_pipeline.py` first to generate the processed datasets."
        )
    if not dev_path.exists():
        raise FileNotFoundError(
            f"Dev data not found: {dev_path}\n"
            "Run `python run_pipeline.py` first to generate the processed datasets."
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
    print("[train_extractor] Computing class weights from training set...")
    class_weights = compute_class_weights(train_dataset, num_classes=3, device=device)
    print(f"[train_extractor] Class weights: {class_weights.tolist()}")

    # ── Model ───────────────────────────────────────────────────────────────
    print(f"[train_extractor] Initialising TextualFeatureExtractor ({model_name})...")
    model = TextualFeatureExtractor(model_name=model_name, num_labels=3)

    # Enable gradient checkpointing for small batch sizes to save memory
    if batch_size <= 8:
        print("[train_extractor] Enabling gradient checkpointing (batch_size <= 8).")
        model.deberta.gradient_checkpointing_enable()

    model.to(device)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda" and use_fp16))

    # ── Loss, optimiser, scheduler ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 2e-5)),
        weight_decay=0.01,
    )

    max_epochs: int = training_cfg.get("max_epochs", 10)
    gradient_accumulation: int = training_cfg.get("gradient_accumulation", 4)
    warmup_steps: int = training_cfg.get("warmup_steps", 200)
    early_stopping_patience: int | None = training_cfg.get("early_stopping_patience", None)
    max_grad_norm: float = 1.0

    if args.patience is not None:
        early_stopping_patience = args.patience

    total_update_steps = (len(train_loader) // gradient_accumulation) * max_epochs
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
            "val_per_class_f1": val_metrics["per_class_f1"],
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

    # ── Save final model ────────────────────────────────────────────────────
    final_checkpoint_path = checkpoints_dir / "final_model.pt"
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"\n[train_extractor] Final model saved to {final_checkpoint_path}")
    print(f"[train_extractor] Best model (macro-F1={best_macro_f1:.4f}) saved to {best_checkpoint_path}")

    # ── Save training metrics ────────────────────────────────────────────────
    metrics_path = outputs_dir / "extractor_train_metrics.json"
    summary = {
        "best_val_macro_f1": best_macro_f1,
        "total_epochs": max_epochs,
        "model_name": model_name,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "device": str(device),
        "best_checkpoint": str(best_checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path),
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
