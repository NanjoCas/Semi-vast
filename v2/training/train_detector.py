"""
train_detector.py (v2)
======================
Phase 4: Train the Dual-Channel Detector for one ablation method.

v2 changes vs. v1:
  - Pseudo-labeled samples are claim+evidence pairs and go through the same
    reasoning channel that validation/test use (v1: claim-only content channel,
    never evaluated).
  - Every method uses the same supervised loss (DualChannelDetector.
    compute_supervised_loss). v1's supervised-only baseline silently used plain
    CE while the others used focal + logit adjustment.
  - Pseudo loss type / weight normalisation / lambda come from config
    (algorithm.pseudo_loss_type, algorithm.pseudo_weight_norm,
    hyperparameters.lambda_init/lambda_final).
  - warmup is a ratio of total optimizer steps (training.warmup_ratio).
  - Paths come from --run_dir and --method; test-set predictions are saved for
    paired significance tests; checkpoints hold model weights only and are
    deleted after evaluation unless --keep_checkpoint is given.

Methods: A (no pseudo labels), B, W, R, C, O (see training/build_baseline_sets.py).

Usage:
    python training/train_detector.py --run_dir runs/r0.10_s42 --method C --seed 42 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

V2_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2_ROOT))

from common.data_utils import ClaimEvidenceDataset, ID2LABEL, NUM_LABELS  # noqa: E402
from common.paths import RunPaths, cfg_path, labeled_split_path, load_config  # noqa: E402
from models.detector import DualChannelDetector, compute_class_priors, compute_class_weights  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_detector")
for noisy_name in ("httpx", "urllib3", "transformers", "huggingface_hub"):
    logging.getLogger(noisy_name).setLevel(logging.WARNING)


# ============================================================================
# Helpers
# ============================================================================

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infinite_cycle(loader: DataLoader) -> Iterator[dict]:
    while True:
        for batch in loader:
            yield batch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summarize_label_distribution(label_ids: list[int], name: str) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for lid in label_ids:
        if int(lid) in counts:
            counts[int(lid)] += 1
    total = sum(counts.values())
    parts = [
        f"{ID2LABEL[lid]}={counts[lid]} ({(100.0 * counts[lid] / total) if total else 0.0:.1f}%)"
        for lid in (0, 1, 2)
    ]
    log.info("%s distribution | %s", name, ", ".join(parts))
    return counts


def _build_balanced_sampler_from_labels(label_ids: list[int], power: float = 1.0) -> WeightedRandomSampler:
    counts = {0: 0, 1: 0, 2: 0}
    for lid in label_ids:
        counts[int(lid)] = counts.get(int(lid), 0) + 1
    per_class = {lid: (0.0 if cnt <= 0 else 1.0 / (float(cnt) ** float(power))) for lid, cnt in counts.items()}
    sample_weights = torch.tensor([per_class[int(lid)] for lid in label_ids], dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(label_ids), replacement=True)


def _to_device(batch: dict, device: torch.device) -> tuple:
    token_type_ids = batch.get("token_type_ids")
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        token_type_ids.to(device) if token_type_ids is not None else None,
    )


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
    return_predictions: bool = False,
):
    """Evaluate on a labeled loader via the reasoning channel (claim + evidence)."""
    detector.eval()
    all_logits, all_labels, all_ids = [], [], []

    for batch in tqdm(loader, desc=split_name, unit="batch", leave=False, dynamic_ncols=True):
        input_ids, attention_mask, token_type_ids = _to_device(batch, device)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(use_amp and device.type == "cuda"),
            dtype=amp_dtype,
        ):
            logits = detector.forward_reasoning(input_ids, attention_mask, token_type_ids)
        all_logits.append(logits.float().cpu())
        all_labels.append(batch["label"].cpu())
        all_ids.extend(batch["id"])

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    probs = torch.softmax(logits_cat, dim=-1).numpy()
    preds = logits_cat.argmax(dim=-1).numpy()
    true = labels_cat.numpy()

    acc = float(accuracy_score(true, preds))
    macro_f1 = float(f1_score(true, preds, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")

    log.info("[%s]  acc=%.4f  macro-F1=%.4f  AUC=%.4f", split_name, acc, macro_f1, auc)
    metrics = {"accuracy": acc, "macro_f1": macro_f1, "auc": auc}
    if not return_predictions:
        return metrics
    predictions = [
        {"id": i, "gold": int(t), "pred": int(p), "probs": [round(float(x), 6) for x in pr]}
        for i, t, p, pr in zip(all_ids, true, preds, probs)
    ]
    return metrics, predictions


# ============================================================================
# Training loop
# ============================================================================

def train(
    detector: DualChannelDetector,
    labeled_loader: DataLoader,
    pseudo_loader: Optional[DataLoader],
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    ckpt_path: Path,
    labeled_class_weights: Optional[torch.Tensor] = None,
    labeled_class_priors: Optional[torch.Tensor] = None,
) -> list[dict]:
    """Joint training: each labeled batch is paired with `pseudo_ratio` pseudo batches."""
    train_cfg = cfg.get("training", {})
    algo_cfg = cfg.get("algorithm", {})

    max_epochs = int(train_cfg.get("max_epochs", 8))
    lr = float(train_cfg.get("learning_rate", 1e-5))
    grad_accum = int(train_cfg.get("gradient_accumulation", 4))
    max_grad_norm = 1.0
    pseudo_ratio = int(train_cfg.get("pseudo_ratio", 3))
    use_pseudo = pseudo_loader is not None and pseudo_ratio > 0

    use_bf16 = bool(train_cfg.get("use_bf16", False))
    use_fp16 = bool(train_cfg.get("use_fp16", False))
    use_amp = bool(use_bf16 or use_fp16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda" and use_fp16))

    loss_kwargs = {
        "class_weights": labeled_class_weights.to(device) if labeled_class_weights is not None else None,
        "class_priors": labeled_class_priors.to(device) if labeled_class_priors is not None else None,
        "loss_type": str(algo_cfg.get("loss_type", "ce")).lower(),
        "focal_gamma": float(algo_cfg.get("focal_gamma", 2.0)),
        "logit_adjust_tau": float(algo_cfg.get("logit_adjust_tau", 0.0)),
    }
    pseudo_loss_type = str(algo_cfg.get("pseudo_loss_type", "ce")).lower()
    pseudo_weight_norm = str(algo_cfg.get("pseudo_weight_norm", "mean")).lower()
    log.info(
        "Loss: sup=%s gamma=%.2f tau=%.2f | pseudo=%s norm=%s | use_pseudo=%s ratio=%d",
        loss_kwargs["loss_type"], loss_kwargs["focal_gamma"], loss_kwargs["logit_adjust_tau"],
        pseudo_loss_type, pseudo_weight_norm, use_pseudo, pseudo_ratio,
    )

    detector.enable_gradient_checkpointing()

    optimizer = AdamW([p for p in detector.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(labeled_loader) / grad_accum)
    total_steps = steps_per_epoch * max_epochs
    if "warmup_ratio" in train_cfg:
        warmup_steps = int(round(float(train_cfg["warmup_ratio"]) * total_steps))
    else:
        warmup_steps = int(train_cfg.get("warmup_steps", 100))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    log.info("steps_per_epoch=%d total_steps=%d warmup_steps=%d", steps_per_epoch, total_steps, warmup_steps)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    pseudo_iter = _infinite_cycle(pseudo_loader) if use_pseudo else None
    best_val_f1 = -1.0
    history: list[dict] = []
    global_step = 0
    skipped_nonfinite = 0

    def optimizer_step() -> None:
        nonlocal global_step
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

    for epoch in range(1, max_epochs + 1):
        detector.train()
        epoch_loss = epoch_sup = epoch_pseudo = 0.0
        n_batches = 0
        optimizer.zero_grad()
        t_epoch = time.time()

        train_iter = tqdm(labeled_loader, desc=f"train epoch {epoch}/{max_epochs}", unit="batch", leave=False, dynamic_ncols=True)
        for labeled_batch in train_iter:
            input_ids, attention_mask, token_type_ids = _to_device(labeled_batch, device)
            labels = labeled_batch["label"].to(device)
            lam = detector.get_lambda(global_step, total_steps)

            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda"), dtype=amp_dtype):
                labeled_logits = detector.forward_reasoning(input_ids, attention_mask, token_type_ids)
                if use_pseudo:
                    pseudo_batches = [next(pseudo_iter) for _ in range(pseudo_ratio)]
                    p_ids = torch.cat([b["input_ids"] for b in pseudo_batches]).to(device)
                    p_mask = torch.cat([b["attention_mask"] for b in pseudo_batches]).to(device)
                    p_tt = torch.cat([b["token_type_ids"] for b in pseudo_batches]).to(device)
                    p_labels = torch.cat([b["label"] for b in pseudo_batches]).to(device)
                    p_weights = torch.cat([b["weight"] for b in pseudo_batches]).to(device).float()
                    # v2: same claim+evidence input and same channel as evaluation.
                    pseudo_logits = detector.forward_reasoning(p_ids, p_mask, p_tt)
                    loss_dict = detector.compute_joint_loss(
                        labeled_logits=labeled_logits,
                        labeled_labels=labels,
                        pseudo_logits=pseudo_logits,
                        pseudo_labels=p_labels,
                        pseudo_weights=p_weights,
                        lambda_val=lam,
                        pseudo_loss_type=pseudo_loss_type,
                        pseudo_weight_norm=pseudo_weight_norm,
                        **loss_kwargs,
                    )
                else:
                    l_sup = detector.compute_supervised_loss(labeled_logits, labels, **loss_kwargs)
                    loss_dict = {"total": l_sup, "supervised": l_sup, "pseudo": torch.zeros((), device=device)}
                loss = loss_dict["total"]

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                log.warning("Non-finite loss at epoch=%d (skipped=%d)", epoch, skipped_nonfinite)
                continue

            loss = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += float(loss_dict["total"].item())
            epoch_sup += float(loss_dict["supervised"].item())
            epoch_pseudo += float(loss_dict["pseudo"].item())
            n_batches += 1
            if n_batches % 10 == 0:
                train_iter.set_postfix(loss=f"{epoch_loss / n_batches:.4f}", lam=f"{lam:.3f}", step=global_step)
            if n_batches % grad_accum == 0:
                optimizer_step()

        if n_batches % grad_accum != 0:
            optimizer_step()

        denom = max(n_batches, 1)
        log.info(
            "Epoch %d/%d loss=%.4f (sup=%.4f pseudo=%.4f) time=%.1fs step=%d",
            epoch, max_epochs, epoch_loss / denom, epoch_sup / denom, epoch_pseudo / denom,
            time.time() - t_epoch, global_step,
        )

        val_metrics = evaluate(detector, val_loader, device, f"val/epoch{epoch}", use_amp, amp_dtype)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / denom,
            "train_sup_loss": epoch_sup / denom,
            "train_pseudo_loss": epoch_pseudo / denom,
            "lambda": detector.get_lambda(global_step, total_steps),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            # Model weights only: optimizer state would triple the file size.
            torch.save({"epoch": epoch, "model_state_dict": detector.state_dict(), "val_metrics": val_metrics}, ckpt_path)
            log.info("  New best val macro-F1=%.4f -> %s", best_val_f1, ckpt_path)

    log.info("Training complete. Best val macro-F1: %.4f", best_val_f1)
    return history


def save_detector_plots(history: list[dict], outputs_dir: Path, test_metrics: Optional[dict] = None) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.plot(epochs, [h["train_sup_loss"] for h in history], marker="o", label="supervised")
    ax1.plot(epochs, [h["train_pseudo_loss"] for h in history], marker="s", label="pseudo")
    ax1.set_title("Detector train loss")
    ax1.legend()
    for ax, key, title in ((ax2, "accuracy", "Accuracy"), (ax3, "macro_f1", "Macro-F1"), (ax4, "auc", "AUC (OvR)")):
        ax.plot(epochs, [h.get(f"val_{key}", float("nan")) for h in history], marker="o", label=f"val {key}")
        if test_metrics is not None and not np.isnan(float(test_metrics.get(key, float("nan")))):
            ax.axhline(float(test_metrics[key]), linestyle=":", color="black", label=f"test {key}")
        ax.set_title(title)
        ax.legend()
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outputs_dir / "detector_training_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(outputs_dir / "detector_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 (v2): train the detector for one ablation method.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["A", "B", "W", "R", "C", "O"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--keep_checkpoint", action="store_true", help="Keep best_model.pt after evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = RunPaths(args.run_dir)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("models", {})
    hyper_cfg = cfg.get("hyperparameters", {})
    imbalance_cfg = cfg.get("imbalance", {})

    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    _set_seed(seed)
    device = torch.device(args.device) if args.device else _resolve_device()
    if device.type == "cuda":
        use_tf32 = bool(train_cfg.get("use_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cudnn.benchmark = True
    log.info("run=%s method=%s seed=%d device=%s", paths.root.name, args.method, seed, device)

    train_path = paths.labeled_train
    dev_path = labeled_split_path(cfg, "dev")
    test_path = labeled_split_path(cfg, "test")
    pseudo_path = paths.pseudo_set(args.method)
    for p in [train_path, dev_path, test_path] + ([pseudo_path] if pseudo_path else []):
        if not p.exists():
            log.error("Required file not found: %s", p)
            sys.exit(1)

    model_name = model_cfg.get("deberta_base", "microsoft/deberta-v3-large")
    model_cache_dir = cfg_path(cfg, "model_cache_dir")
    local_only = bool(cfg.get("use_local_models", False))
    cache_kwargs = {"cache_dir": str(model_cache_dir)}
    if local_only:
        cache_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_kwargs)

    max_length = int(train_cfg.get("max_length", 384))
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }

    train_dataset = ClaimEvidenceDataset(str(train_path), tokenizer, max_length=max_length)
    dev_dataset = ClaimEvidenceDataset(str(dev_path), tokenizer, max_length=max_length)
    test_dataset = ClaimEvidenceDataset(str(test_path), tokenizer, max_length=max_length)

    labeled_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(dev_dataset, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)
    _summarize_label_distribution(train_dataset.label_ids, "Labeled train")

    pseudo_loader = None
    pseudo_size = 0
    if pseudo_path is not None:
        pseudo_dataset = ClaimEvidenceDataset(str(pseudo_path), tokenizer, max_length=max_length)
        pseudo_size = len(pseudo_dataset)
        if pseudo_size == 0:
            # Can legitimately happen (e.g. no sample clears the confidence threshold
            # at a low label ratio). Record it instead of failing the whole matrix.
            out_dir = paths.detector_outputs(args.method)
            out_dir.mkdir(parents=True, exist_ok=True)
            reason = f"pseudo set for method {args.method} is empty: {pseudo_path}"
            with open(out_dir / "skipped.json", "w", encoding="utf-8") as fh:
                json.dump({"run": paths.root.name, "method": args.method, "seed": seed, "reason": reason}, fh, indent=2)
            log.warning("SKIPPED: %s", reason)
            return
        _summarize_label_distribution(pseudo_dataset.label_ids, f"Pseudo set {args.method}")
        sampler = None
        if bool(imbalance_cfg.get("use_balanced_pseudo_sampler", True)):
            sampler = _build_balanced_sampler_from_labels(
                pseudo_dataset.label_ids, power=float(imbalance_cfg.get("pseudo_sampler_power", 1.0))
            )
        pseudo_loader = DataLoader(
            pseudo_dataset,
            batch_size=int(train_cfg.get("pseudo_batch_size", batch_size)),
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=pseudo_size >= int(train_cfg.get("pseudo_batch_size", batch_size)),
            **loader_kwargs,
        )

    detector = DualChannelDetector(
        model_name=model_name,
        num_labels=NUM_LABELS,
        lambda_init=float(hyper_cfg.get("lambda_init", 0.1)),
        lambda_final=float(hyper_cfg.get("lambda_final", 0.3)),
        cache_dir=str(model_cache_dir),
        local_files_only=local_only,
    ).to(device).float()

    ckpt_path = paths.detector_ckpt_dir(args.method) / "best_model.pt"
    history = train(
        detector=detector,
        labeled_loader=labeled_loader,
        pseudo_loader=pseudo_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        ckpt_path=ckpt_path,
        labeled_class_weights=compute_class_weights(str(train_path)),
        labeled_class_priors=compute_class_priors(str(train_path)),
    )

    out_dir = paths.detector_outputs(args.method)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)

    ckpt = torch.load(ckpt_path, map_location=device)
    detector.load_state_dict(ckpt["model_state_dict"])
    use_bf16 = bool(train_cfg.get("use_bf16", False))
    test_metrics, predictions = evaluate(
        detector, test_loader, device, "test",
        use_amp=bool(use_bf16 or train_cfg.get("use_fp16", False)),
        amp_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        return_predictions=True,
    )

    results = {
        "run": paths.root.name,
        "method": args.method,
        "seed": seed,
        "best_epoch": ckpt["epoch"],
        "best_val": ckpt["val_metrics"],
        "pseudo_set": str(pseudo_path) if pseudo_path else None,
        "pseudo_size": pseudo_size,
        "labeled_size": len(train_dataset),
        **test_metrics,
    }
    with open(out_dir / "test_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    with open(out_dir / "test_predictions.jsonl", "w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        save_detector_plots(history, out_dir, test_metrics)
    except Exception as e:
        log.warning("Plot generation skipped due to error: %s", e)

    if not args.keep_checkpoint and bool(cfg.get("experiment", {}).get("cleanup_checkpoints", True)):
        shutil.rmtree(ckpt_path.parent, ignore_errors=True)
        log.info("Removed checkpoint dir %s (cleanup_checkpoints=true)", ckpt_path.parent)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
