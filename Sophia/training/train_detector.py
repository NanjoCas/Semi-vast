"""
train_detector.py
=================
Phase 4: Joint training of the Dual-Channel Detector.

Two input channels are trained jointly:
    - Reasoning channel : labeled samples  (ClaimEvidenceDataset, 1x)
    - Content channel   : pseudo-labeled samples (claim-only, 3x)

The two DataLoaders are cycled in a 1:3 ratio per step.
Validation is run after every epoch and the best macro-F1 checkpoint is kept.
A final evaluation on the held-out test set is performed after training.

Usage:
    python training/train_detector.py --config configs/config.yaml
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project-root imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_pipeline import ClaimEvidenceDataset                   # noqa: E402
from models.detector import DualChannelDetector, compute_class_weights, compute_class_priors  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_detector")

# Keep output readable during long runs.
for noisy_name in ("httpx", "urllib3", "transformers", "huggingface_hub"):
    logging.getLogger(noisy_name).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
NUM_LABELS = 3


# ============================================================================
# Device resolution
# ============================================================================

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Pseudo-labeled content-channel Dataset
# ============================================================================

class PseudoClaimDataset(Dataset):
    """
    Content-channel dataset for pseudo-labeled samples.

    Each record is expected to contain at least a "claim" field and either a
    "pseudo_label" (int 0/1/2) or a "label" (str) field.

    Tokenisation uses claim text only (no evidence) at max_length=256 to keep
    content-channel batches fast.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: list[dict] = []

        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        # Cache resolved pseudo labels once for sampling diagnostics/samplers.
        self.label_ids: list[int] = []
        for rec in self.records:
            raw_label = rec.get("pseudo_label")
            if isinstance(raw_label, int) and 0 <= raw_label < NUM_LABELS:
                label_id = int(raw_label)
            elif isinstance(raw_label, str):
                label_id = LABEL2ID.get(raw_label, 2)
            else:
                label_id = LABEL2ID.get(rec.get("label", "NOT_ENOUGH_INFO"), 2)
            self.label_ids.append(label_id)

        log.info(
            "PseudoClaimDataset: loaded %d records from %s",
            len(self.records),
            jsonl_path,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        claim: str = rec.get("claim", "")

        # Resolve pseudo label
        raw_label = rec.get("pseudo_label")
        if isinstance(raw_label, int) and 0 <= raw_label < NUM_LABELS:
            label_id = raw_label
        elif isinstance(raw_label, str):
            label_id = LABEL2ID.get(raw_label, 2)
        else:
            label_id = LABEL2ID.get(rec.get("label", "NOT_ENOUGH_INFO"), 2)

        encoding = self.tokenizer(
            claim,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        weight = float(rec.get("weight", 1.0))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids",
                torch.zeros(self.max_length, dtype=torch.long),
            ).squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long),
            "weight": torch.tensor(weight, dtype=torch.float),
            "id": rec.get("id", ""),
        }


# ============================================================================
# Helpers
# ============================================================================

def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _infinite_cycle(loader: DataLoader) -> Iterator[dict]:
    """Yield batches from *loader* in an infinite loop."""
    while True:
        for batch in loader:
            yield batch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _label_name(label_id: int) -> str:
    return {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO"}.get(int(label_id), str(label_id))


def _summarize_label_distribution(label_ids: list[int], name: str) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for lid in label_ids:
        if int(lid) in counts:
            counts[int(lid)] += 1
    total = sum(counts.values())
    parts = []
    for lid in (0, 1, 2):
        cnt = counts[lid]
        pct = (100.0 * cnt / total) if total else 0.0
        parts.append(f"{_label_name(lid)}={cnt} ({pct:.1f}%)")
    max_cnt = max(counts.values()) if total else 0
    min_nonzero = min([c for c in counts.values() if c > 0], default=0)
    ratio = (max_cnt / min_nonzero) if min_nonzero > 0 else float('inf')
    log.info("%s distribution | %s | imbalance_ratio(max/min_nonzero)=%.2f", name, ", ".join(parts), ratio)
    return counts


def _build_balanced_sampler_from_labels(label_ids: list[int], power: float = 1.0) -> WeightedRandomSampler:
    counts = {0: 0, 1: 0, 2: 0}
    for lid in label_ids:
        counts[int(lid)] = counts.get(int(lid), 0) + 1

    per_class = {}
    for lid, cnt in counts.items():
        per_class[lid] = 0.0 if cnt <= 0 else 1.0 / (float(cnt) ** float(power))

    sample_weights = torch.tensor([per_class[int(lid)] for lid in label_ids], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(label_ids),
        replacement=True,
    )
    return sampler


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(
    detector: DualChannelDetector,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "val",
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """
    Evaluate the detector on a labeled DataLoader.

    Returns a dict with keys: accuracy, macro_f1, auc.
    """
    detector.eval()

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    eval_iter = tqdm(
        loader,
        desc=f"{split_name}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in eval_iter:
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
            logits = detector.forward_reasoning(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_cat = torch.cat(all_logits, dim=0)       # (N, C)
    labels_cat = torch.cat(all_labels, dim=0)       # (N,)

    probs = torch.softmax(logits_cat.float(), dim=-1).numpy()
    preds = logits_cat.argmax(dim=-1).numpy()
    true = labels_cat.numpy()

    acc = float(accuracy_score(true, preds))
    macro_f1 = float(f1_score(true, preds, average="macro", zero_division=0))

    # AUC: one-vs-rest macro for multi-class
    try:
        auc = float(roc_auc_score(true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        # Fallback when a class is absent from the batch
        auc = float("nan")

    log.info(
        "[%s]  acc=%.4f  macro-F1=%.4f  AUC=%.4f",
        split_name,
        acc,
        macro_f1,
        auc,
    )
    return {"accuracy": acc, "macro_f1": macro_f1, "auc": auc}


# ============================================================================
# Training loop
# ============================================================================

def train(
    detector: DualChannelDetector,
    labeled_loader: DataLoader,
    pseudo_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    project_root: Path,
    labeled_class_weights: torch.Tensor | None = None,
    labeled_class_priors: torch.Tensor | None = None,
) -> list[dict]:
    """
    Joint training loop with 1:3 labeled-to-pseudo batch ratio.

    Returns:
        list[dict]: per-epoch training metrics (loss, val accuracy, val macro_f1, val auc).
    """
    train_cfg = cfg.get("training", {})
    max_epochs: int = train_cfg.get("max_epochs", 10)
    lr: float = train_cfg.get("learning_rate", 2e-5)
    warmup_steps: int = train_cfg.get("warmup_steps", 200)
    grad_accum: int = train_cfg.get("gradient_accumulation", 4)
    max_grad_norm: float = 1.0
    pseudo_ratio: int = 3  # pseudo batches per labeled batch
    use_bf16 = bool(train_cfg.get("use_bf16", False))
    use_fp16 = bool(train_cfg.get("use_fp16", False))
    use_amp = bool(use_bf16 or use_fp16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda" and use_fp16))

    algo_cfg = cfg.get("algorithm", {})
    loss_type: str = str(algo_cfg.get("loss_type", "ce")).lower()
    focal_gamma: float = float(algo_cfg.get("focal_gamma", 2.0))
    logit_adjust_tau: float = float(algo_cfg.get("logit_adjust_tau", 0.0))

    class_weights_device = None
    if labeled_class_weights is not None:
        class_weights_device = labeled_class_weights.to(device)
        log.info("Using supervised class weights: %s", [round(float(x), 4) for x in class_weights_device.detach().cpu().tolist()])

    class_priors_device = None
    if labeled_class_priors is not None:
        class_priors_device = labeled_class_priors.to(device)
        log.info("Using class priors: %s", [round(float(x), 4) for x in class_priors_device.detach().cpu().tolist()])

    log.info(
        "Algorithm loss settings: loss_type=%s focal_gamma=%.2f logit_adjust_tau=%.3f",
        loss_type,
        focal_gamma,
        logit_adjust_tau,
    )

    # -- Enable gradient checkpointing (memory efficiency) --
    if hasattr(detector, "gradient_checkpointing_enable"):
        detector.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled.")
    else:
        # Try to enable on the underlying backbone if accessible
        for attr_name in ("reasoning_encoder", "content_encoder", "deberta", "backbone"):
            backbone = getattr(detector, attr_name, None)
            if backbone is not None and hasattr(backbone, "gradient_checkpointing_enable"):
                backbone.gradient_checkpointing_enable()
                log.info("Gradient checkpointing enabled on '%s'.", attr_name)
                break

    # -- Optimizer --
    optimizer = AdamW(
        [p for p in detector.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    # Estimate total optimizer steps
    steps_per_epoch = math.ceil(len(labeled_loader) / grad_accum)
    total_steps = steps_per_epoch * max_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # -- Checkpoint dir --
    ckpt_dir = project_root / "checkpoints" / "detector"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_model.pt"

    best_val_f1: float = -1.0
    history: list[dict] = []
    global_step: int = 0
    skipped_nonfinite_batches: int = 0

    pseudo_iter = _infinite_cycle(pseudo_loader)

    log.info(
        "Starting training: max_epochs=%d, steps_per_epoch≈%d, total_steps≈%d",
        max_epochs,
        steps_per_epoch,
        total_steps,
    )

    for epoch in range(1, max_epochs + 1):
        detector.train()
        epoch_loss: float = 0.0
        n_accum_batches: int = 0
        optimizer.zero_grad()

        t_epoch = time.time()

        train_iter = tqdm(
            labeled_loader,
            desc=f"train epoch {epoch}/{max_epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for labeled_batch in train_iter:
            # ---- Forward: labeled (reasoning channel) ----
            input_ids = labeled_batch["input_ids"].to(device)
            attention_mask = labeled_batch["attention_mask"].to(device)
            token_type_ids = labeled_batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = labeled_batch["label"].to(device)

            lam = detector.get_lambda(global_step, total_steps)

            # Collect 3 pseudo batches
            pseudo_batches = [next(pseudo_iter) for _ in range(pseudo_ratio)]

            # Merge pseudo mini-batches for one content-channel forward pass.
            pseudo_input_ids = torch.cat([b["input_ids"] for b in pseudo_batches], dim=0).to(device)
            pseudo_attention_mask = torch.cat([b["attention_mask"] for b in pseudo_batches], dim=0).to(device)
            if all("token_type_ids" in b for b in pseudo_batches):
                pseudo_token_type_ids = torch.cat([b["token_type_ids"] for b in pseudo_batches], dim=0).to(device)
            else:
                pseudo_token_type_ids = None
            pseudo_labels = torch.cat([b["label"] for b in pseudo_batches], dim=0).to(device)
            pseudo_weights = torch.cat([
                b.get("weight", torch.ones_like(b["label"], dtype=torch.float))
                for b in pseudo_batches
            ], dim=0).to(device).float()
            pseudo_weights = torch.nan_to_num(pseudo_weights, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(use_amp and device.type == "cuda"),
                dtype=amp_dtype,
            ):
                labeled_logits = detector.forward_reasoning(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                pseudo_logits = detector.forward_content(
                    input_ids=pseudo_input_ids,
                    attention_mask=pseudo_attention_mask,
                    token_type_ids=pseudo_token_type_ids,
                )
                loss_dict = detector.compute_joint_loss(
                    labeled_logits=labeled_logits,
                    labeled_labels=labels,
                    pseudo_logits=pseudo_logits,
                    pseudo_labels=pseudo_labels,
                    pseudo_weights=pseudo_weights,
                    lambda_val=lam,
                    class_weights=class_weights_device,
                    loss_type=loss_type,
                    focal_gamma=focal_gamma,
                    logit_adjust_tau=logit_adjust_tau,
                    class_priors=class_priors_device,
                )
                loss = loss_dict["total"]

            if not torch.isfinite(loss):
                skipped_nonfinite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if skipped_nonfinite_batches <= 5 or skipped_nonfinite_batches % 20 == 0:
                    log.warning(
                        "Non-finite loss detected at epoch=%d batch=%d (skipped=%d)",
                        epoch,
                        n_accum_batches + 1,
                        skipped_nonfinite_batches,
                    )
                continue

            # Scale for gradient accumulation
            loss = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * grad_accum
            n_accum_batches += 1

            if n_accum_batches % 10 == 0:
                train_iter.set_postfix(
                    loss=f"{(epoch_loss / max(n_accum_batches, 1)):.4f}",
                    lam=f"{lam:.5f}",
                    step=global_step,
                )

            if n_accum_batches % grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(detector.parameters(), max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # Handle leftover accumulated gradients at end of epoch
        if n_accum_batches % grad_accum != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / max(n_accum_batches, 1)
        elapsed = time.time() - t_epoch

        log.info(
            "Epoch %d/%d  loss=%.4f  time=%.1fs  step=%d  lambda=%.5f  skipped_nonfinite=%d",
            epoch,
            max_epochs,
            avg_loss,
            elapsed,
            global_step,
            detector.get_lambda(global_step, total_steps),
            skipped_nonfinite_batches,
        )

        # ---- Validation ----
        val_metrics = evaluate(
            detector,
            val_loader,
            device,
            split_name=f"val/epoch{epoch}",
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_record)

        # ---- Save best checkpoint ----
        val_f1 = val_metrics["macro_f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": detector.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_f1,
                    "val_metrics": val_metrics,
                },
                best_ckpt_path,
            )
            log.info(
                "  New best val macro-F1=%.4f — checkpoint saved to %s",
                val_f1,
                best_ckpt_path,
            )

    log.info("Training complete. Best val macro-F1: %.4f", best_val_f1)
    return history


def save_detector_plots(
    history: list[dict],
    outputs_dir: Path,
    test_metrics: dict[str, float] | None = None,
) -> list[Path]:
    """Save publication-ready detector training plots (PNG + PDF)."""
    if not history:
        return []

    outputs_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", float("nan")) for h in history]
    val_acc = [h.get("val_accuracy", float("nan")) for h in history]
    val_f1 = [h.get("val_macro_f1", float("nan")) for h in history]
    val_auc = [h.get("val_auc", float("nan")) for h in history]

    created: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.plot(epochs, train_loss, marker="o", linewidth=2, color="tab:blue")
    ax1.set_title("Detector Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(epochs, val_acc, marker="o", linewidth=2, color="tab:green", label="Val Accuracy")
    if test_metrics is not None and "accuracy" in test_metrics:
        ax2.axhline(float(test_metrics["accuracy"]), color="tab:green", linestyle=":", label="Test Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    ax3.plot(epochs, val_f1, marker="o", linewidth=2, color="tab:red", label="Val Macro-F1")
    if test_metrics is not None and "macro_f1" in test_metrics:
        ax3.axhline(float(test_metrics["macro_f1"]), color="tab:red", linestyle=":", label="Test Macro-F1")
    ax3.set_title("Macro-F1")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Macro-F1")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend()

    ax4.plot(epochs, val_auc, marker="o", linewidth=2, color="tab:purple", label="Val AUC")
    if test_metrics is not None and "auc" in test_metrics and not np.isnan(float(test_metrics["auc"])):
        ax4.axhline(float(test_metrics["auc"]), color="tab:purple", linestyle=":", label="Test AUC")
    ax4.set_title("AUC (OvR)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("AUC")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend()

    fig.tight_layout()
    curve_png = outputs_dir / "detector_training_curves.png"
    curve_pdf = outputs_dir / "detector_training_curves.pdf"
    fig.savefig(curve_png, dpi=300, bbox_inches="tight")
    fig.savefig(curve_pdf, bbox_inches="tight")
    plt.close(fig)
    created.extend([curve_png, curve_pdf])

    return created


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: Joint training of the Dual-Channel Detector."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cpu / cuda / mps). Auto-detected when omitted.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
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
    # Seed + device
    # ------------------------------------------------------------------
    seed: int = cfg.get("training", {}).get("seed", 42)
    _set_seed(seed)

    device = torch.device(args.device) if args.device else _resolve_device()
    log.info("Using device: %s", device)
    train_cfg_for_amp = cfg.get("training", {})
    use_tf32 = bool(train_cfg_for_amp.get("use_tf32", True))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cudnn.benchmark = True
        log.info("CUDA TF32 enabled: %s", use_tf32)

    # ------------------------------------------------------------------
    # Resolve data paths
    # ------------------------------------------------------------------
    labeled_dir = project_root / "processed" / "labeled"
    pseudo_dir = project_root / "processed" / "pseudo_labels"

    train_path = labeled_dir / "train.jsonl"
    dev_path = labeled_dir / "dev.jsonl"
    test_path = labeled_dir / "test.jsonl"

    for p in (train_path, dev_path, test_path):
        if not p.exists():
            log.error("Required labeled data file not found: %s", p)
            sys.exit(1)

    # RL-selected pseudo labels with fallback
    rl_selected_path = pseudo_dir / "rl_selected.jsonl"
    pseudo_path: Path
    if rl_selected_path.exists():
        pseudo_path = rl_selected_path
        log.info("Using RL-selected pseudo labels: %s", pseudo_path)
    else:
        filtered_path = pseudo_dir / "pseudo_labeled_filtered.jsonl"
        if filtered_path.exists():
            pseudo_path = filtered_path
            log.warning(
                "rl_selected.jsonl not found; falling back to %s", pseudo_path
            )
        else:
            unfiltered_path = pseudo_dir / "pseudo_labeled.jsonl"
            if unfiltered_path.exists():
                pseudo_path = unfiltered_path
                log.warning(
                    "Neither rl_selected nor filtered pseudo labels found; "
                    "falling back to %s",
                    pseudo_path,
                )
            else:
                log.error(
                    "No pseudo-label file found. Searched:\n  %s\n  %s\n  %s",
                    rl_selected_path,
                    filtered_path,
                    unfiltered_path,
                )
                sys.exit(1)

    # ------------------------------------------------------------------
    # Training hyper-parameters
    # ------------------------------------------------------------------
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("models", {})

    labeled_batch_size: int = train_cfg.get("batch_size", 16)
    # We already sample `pseudo_ratio` batches per step in the training loop.
    # Keep each pseudo mini-batch the same size as labeled to realize a 1:3 ratio.
    pseudo_batch_size: int = labeled_batch_size
    max_length: int = train_cfg.get("max_length", 512)
    num_workers: int = int(train_cfg.get("num_workers", 0))
    deberta_model_name: str = model_cfg.get("deberta_base", "microsoft/deberta-v3-base")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    log.info("Loading tokenizer: %s", deberta_model_name)
    tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    log.info("Building labeled DataLoaders …")
    train_dataset = ClaimEvidenceDataset(
        str(train_path),
        tokenizer,
        max_length=max_length,
    )
    dev_dataset = ClaimEvidenceDataset(
        str(dev_path),
        tokenizer,
        max_length=max_length,
    )
    test_dataset = ClaimEvidenceDataset(
        str(test_path),
        tokenizer,
        max_length=max_length,
    )

    imbalance_cfg = cfg.get("imbalance", {})
    use_balanced_pseudo_sampler = bool(imbalance_cfg.get("use_balanced_pseudo_sampler", True))
    pseudo_sampler_power = float(imbalance_cfg.get("pseudo_sampler_power", 1.0))

    labeled_loader = DataLoader(
        train_dataset,
        batch_size=labeled_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        dev_dataset,
        batch_size=labeled_batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=labeled_batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    log.info("Building pseudo-label DataLoader (content channel) …")
    pseudo_dataset = PseudoClaimDataset(
        str(pseudo_path),
        tokenizer,
        max_length=256,
    )
    pseudo_sampler = None
    if use_balanced_pseudo_sampler:
        pseudo_sampler = _build_balanced_sampler_from_labels(
            pseudo_dataset.label_ids,
            power=pseudo_sampler_power,
        )
        log.info(
            "Enabled balanced pseudo sampler: power=%.2f replacement=True samples=%d",
            pseudo_sampler_power,
            len(pseudo_dataset.label_ids),
        )

    pseudo_loader = DataLoader(
        pseudo_dataset,
        batch_size=pseudo_batch_size,
        shuffle=(pseudo_sampler is None),
        sampler=pseudo_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    log.info(
        "DataLoaders ready: labeled_train=%d batches, pseudo=%d batches, val=%d batches",
        len(labeled_loader),
        len(pseudo_loader),
        len(val_loader),
    )

    # -- Class distribution diagnostics --
    train_labels = [int(r.get("label", 2)) if isinstance(r.get("label"), int) else LABEL2ID.get(r.get("label", "NOT_ENOUGH_INFO"), 2) for r in train_dataset.records]
    _summarize_label_distribution(train_labels, "Labeled train")
    _summarize_label_distribution(pseudo_dataset.label_ids, "Pseudo selected")

    # -- Supervised class weights/priors from labeled train --
    labeled_class_weights = compute_class_weights(str(train_path))
    labeled_class_priors = compute_class_priors(str(train_path))

    # ------------------------------------------------------------------
    # Initialise DualChannelDetector
    # ------------------------------------------------------------------
    log.info("Initialising DualChannelDetector …")
    hyper_cfg = cfg.get("hyperparameters", {})
    detector = DualChannelDetector(
        model_name=deberta_model_name,
        num_labels=NUM_LABELS,
        lambda_init=hyper_cfg.get("lambda_init", 0.1),
        lambda_final=hyper_cfg.get("lambda", 0.3),
    )
    detector.to(device)

    # Optionally resume
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            log.info("Resuming from checkpoint: %s", resume_path)
            ckpt = torch.load(resume_path, map_location=device)
            detector.load_state_dict(ckpt["model_state_dict"])
        else:
            log.warning("Resume checkpoint not found: %s — starting fresh.", resume_path)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    history = train(
        detector=detector,
        labeled_loader=labeled_loader,
        pseudo_loader=pseudo_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        project_root=project_root,
        labeled_class_weights=labeled_class_weights,
        labeled_class_priors=labeled_class_priors,
    )

    # ------------------------------------------------------------------
    # Save training curve
    # ------------------------------------------------------------------
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    train_metrics_path = outputs_dir / "detector_train_metrics.json"

    log.info("Saving training curve to %s …", train_metrics_path)
    with open(train_metrics_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Final test evaluation (load best checkpoint)
    # ------------------------------------------------------------------
    best_ckpt_path = project_root / "checkpoints" / "detector" / "best_model.pt"
    if best_ckpt_path.exists():
        log.info("Loading best checkpoint for final test evaluation: %s", best_ckpt_path)
        ckpt = torch.load(best_ckpt_path, map_location=device)
        detector.load_state_dict(ckpt["model_state_dict"])
    else:
        log.warning(
            "Best checkpoint not found at %s — evaluating with last epoch weights.",
            best_ckpt_path,
        )

    log.info("Running final evaluation on test set …")
    use_bf16 = bool(cfg.get("training", {}).get("use_bf16", False))
    use_fp16 = bool(cfg.get("training", {}).get("use_fp16", False))
    test_metrics = evaluate(
        detector,
        test_loader,
        device,
        split_name="test",
        use_amp=bool(use_bf16 or use_fp16),
        amp_dtype=(torch.bfloat16 if use_bf16 else torch.float16),
    )

    test_results = {
        "checkpoint": str(best_ckpt_path) if best_ckpt_path.exists() else "last_epoch",
        **test_metrics,
    }

    final_results_path = outputs_dir / "final_test_results.json"
    log.info("Saving final test results to %s …", final_results_path)
    with open(final_results_path, "w", encoding="utf-8") as fh:
        json.dump(test_results, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Save publication-ready plots
    # ------------------------------------------------------------------
    try:
        figure_paths = save_detector_plots(history, outputs_dir, test_metrics=test_metrics)
        if figure_paths:
            log.info("Saved detector plots:")
            for fp in figure_paths:
                log.info("  - %s", fp)
    except Exception as e:
        log.warning("Plot generation skipped due to error: %s", e)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Dual-Channel Detector — Training Summary")
    print("=" * 60)
    print(f"  Epochs trained     : {len(history)}")
    if history:
        best_epoch = max(history, key=lambda x: x.get("val_macro_f1", 0.0))
        print(f"  Best val macro-F1  : {best_epoch.get('val_macro_f1', 0.0):.4f}  (epoch {best_epoch['epoch']})")
        print(f"  Best val accuracy  : {best_epoch.get('val_accuracy', 0.0):.4f}")
        print(f"  Best val AUC       : {best_epoch.get('val_auc', float('nan')):.4f}")
    print("\n  Test set results:")
    print(f"    accuracy   : {test_metrics.get('accuracy', 0.0):.4f}")
    print(f"    macro-F1   : {test_metrics.get('macro_f1', 0.0):.4f}")
    print(f"    AUC        : {test_metrics.get('auc', float('nan')):.4f}")
    print("=" * 60)
    print(f"\n  Training curve     : {train_metrics_path}")
    print(f"  Final test results : {final_results_path}")
    print(f"  Best model ckpt    : {best_ckpt_path}\n")

    print("Phase 4 complete.")


if __name__ == "__main__":
    main()
