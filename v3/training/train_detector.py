"""
train_detector.py (v3)
======================
Phase 4: train the Dual-Channel Detector for one ablation method.

v3 changes vs. v2 (README_v3 section 3):
  - M1  The encoder is initialised from the NLI cross-encoder and its first
        `training.detector_freeze_layers` layers are frozen, exactly like the
        extractor (`training.detector_init_from_nli`). In v2 the detector started
        from raw deberta-v3-large, so the supervised baseline A was weaker than the
        extractor and part of B - A was NLI initialisation leaking in through
        pseudo labels.
  - M2  Fixed training budget for every method: `training.detector_total_steps`
        optimizer steps; each step = 1 labeled batch + `pseudo_ratio` pseudo
        batches (both cycled). LR warmup/decay and the lambda ramp therefore are
        identical across methods. The dev set is evaluated every
        `training.eval_every` steps and the best step is kept. (v2 tied the
        budget to the pseudo-set size: 224-648 steps, and small-set methods
        "took off" at a random point.)
  - M3  Supervised loss = class-weighted CE with the extractor's weights
        (`imbalance.class_weight_power`, `normalize_class_weights`); no focal
        loss and no logit adjustment in the v3 config. Pseudo loss = CE with the
        balanced pseudo sampler, no logit adjustment.
  - M5  The best weights are kept in CPU memory instead of being written to disk
        at every improvement; the labeled loader drops its last partial batch.
  - M6  --train_seed (default: --seed) controls initialisation, dropout and data
        order, so a split can be retrained to measure training noise. Results go
        to outputs/detector/<method> when train_seed == seed and to
        outputs/detector/<method>_t<train_seed> otherwise.

Every method uses the same supervised loss and the same claim+evidence reasoning
channel for labeled and pseudo-labeled pairs (as in v2).

Usage:
    python training/train_detector.py --run_dir runs/r0.10_s42 --method Q --seed 42 --device cuda
    python training/train_detector.py --run_dir runs/r0.10_s42 --method A --seed 42 --train_seed 1042
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data_utils import (  # noqa: E402
    ClaimEvidenceDataset,
    DynamicPaddingCollator,
    ID2LABEL,
    NUM_LABELS,
    class_weights_from_labels,
    pad_sequences,
)
from common.fingerprint import config_fingerprint  # noqa: E402
from common.paths import ALL_METHODS, RunPaths, cfg_path, detector_tag, labeled_split_path, load_config  # noqa: E402
from models.detector import DualChannelDetector, compute_class_priors  # noqa: E402

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
    """Endless iterator; every pass re-shuffles (or re-samples, for the balanced sampler)."""
    while True:
        for batch in loader:
            yield batch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _label_counts(label_ids) -> list[int]:
    counts = Counter(int(x) for x in label_ids)
    return [int(counts.get(k, 0)) for k in range(NUM_LABELS)]


def _summarize_label_distribution(label_ids: list[int], name: str) -> list[int]:
    counts = _label_counts(label_ids)
    total = sum(counts)
    parts = [f"{ID2LABEL[k]}={counts[k]} ({(100.0 * counts[k] / total) if total else 0.0:.1f}%)" for k in range(NUM_LABELS)]
    log.info("%s distribution | %s", name, ", ".join(parts))
    return counts


def _build_balanced_sampler_from_labels(label_ids: list[int], power: float = 1.0) -> WeightedRandomSampler:
    counts = {0: 0, 1: 0, 2: 0}
    for lid in label_ids:
        counts[int(lid)] = counts.get(int(lid), 0) + 1
    per_class = {lid: (0.0 if cnt <= 0 else 1.0 / (float(cnt) ** float(power))) for lid, cnt in counts.items()}
    sample_weights = torch.tensor([per_class[int(lid)] for lid in label_ids], dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(label_ids), replacement=True)


def _concat_batches(batches: list[dict], pad_token_id: int) -> dict:
    """Concatenate pseudo batches whose sequence lengths may differ (dynamic padding)."""
    return {
        "input_ids": pad_sequences([b["input_ids"] for b in batches], pad_token_id, multiple_of=1),
        "attention_mask": pad_sequences([b["attention_mask"] for b in batches], 0, multiple_of=1),
        "token_type_ids": pad_sequences([b["token_type_ids"] for b in batches], 0, multiple_of=1),
        "label": torch.cat([b["label"] for b in batches]),
        "weight": torch.cat([b["weight"] for b in batches]),
    }


def _to_device(batch: dict, device: torch.device) -> tuple:
    token_type_ids = batch.get("token_type_ids")
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        token_type_ids.to(device) if token_type_ids is not None else None,
    )


def training_budget(train_cfg: dict) -> tuple[int, int]:
    """(total optimizer steps, evaluation interval) from the config."""
    if "detector_total_steps" not in train_cfg:
        raise KeyError("training.detector_total_steps is required in v3 (fixed training budget, README_v3 3.2)")
    total_steps = int(train_cfg["detector_total_steps"])
    eval_every = int(train_cfg.get("eval_every", 40))
    if total_steps <= 0 or eval_every <= 0:
        raise ValueError(f"detector_total_steps and eval_every must be positive, got {total_steps}, {eval_every}")
    return total_steps, eval_every


def eval_steps(total_steps: int, eval_every: int) -> list[int]:
    """Steps at which the dev set is evaluated: every eval_every steps, plus the last step."""
    steps = list(range(eval_every, total_steps + 1, eval_every))
    if not steps or steps[-1] != total_steps:
        steps.append(total_steps)
    return steps


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
# Training loop (fixed number of optimizer steps)
# ============================================================================

def train(
    detector: DualChannelDetector,
    labeled_loader: DataLoader,
    pseudo_loader: Optional[DataLoader],
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    class_priors: Optional[torch.Tensor] = None,
    pad_token_id: int = 0,
) -> tuple[list[dict], dict]:
    """
    Train for exactly `training.detector_total_steps` optimizer steps.

    Each optimizer step accumulates `gradient_accumulation` micro-steps; each micro-step is one
    labeled batch plus `pseudo_ratio` pseudo batches (none for method A). Both loaders are cycled,
    so a small pseudo set is simply revisited more often. The dev set is evaluated at
    eval_steps(); the best weights (by dev macro-F1) are copied to CPU memory.

    Returns (history, best) where best = {"step", "val_metrics", "state"}.
    """
    train_cfg = cfg.get("training", {})
    algo_cfg = cfg.get("algorithm", {})

    total_steps, eval_every = training_budget(train_cfg)
    evals = set(eval_steps(total_steps, eval_every))
    grad_accum = max(1, int(train_cfg.get("gradient_accumulation", 1)))
    lr = float(train_cfg.get("learning_rate", 1e-5))
    max_grad_norm = 1.0
    pseudo_ratio = int(train_cfg.get("pseudo_ratio", 3))
    use_pseudo = pseudo_loader is not None and pseudo_ratio > 0

    use_bf16 = bool(train_cfg.get("use_bf16", False))
    use_fp16 = bool(train_cfg.get("use_fp16", False))
    use_amp = bool(use_bf16 or use_fp16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda" and use_fp16))

    loss_kwargs = {
        "class_weights": class_weights.to(device) if class_weights is not None else None,
        "class_priors": class_priors.to(device) if class_priors is not None else None,
        "loss_type": str(algo_cfg.get("loss_type", "ce")).lower(),
        "focal_gamma": float(algo_cfg.get("focal_gamma", 2.0)),
        "logit_adjust_tau": float(algo_cfg.get("logit_adjust_tau", 0.0)),
    }
    pseudo_loss_type = str(algo_cfg.get("pseudo_loss_type", "ce")).lower()
    pseudo_weight_norm = str(algo_cfg.get("pseudo_weight_norm", "mean")).lower()

    if bool(train_cfg.get("gradient_checkpointing", True)):
        detector.enable_gradient_checkpointing()

    trainable = [p for p in detector.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=0.01)
    if "warmup_ratio" in train_cfg:
        warmup_steps = int(round(float(train_cfg["warmup_ratio"]) * total_steps))
    else:
        warmup_steps = int(train_cfg.get("warmup_steps", 0))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    log.info(
        "budget: total_steps=%d eval_every=%d grad_accum=%d warmup_steps=%d lr=%.1e | "
        "per step: %d labeled + %d pseudo pairs | encoder=%s frozen_layers=%d trainable_params=%.1fM",
        total_steps, eval_every, grad_accum, warmup_steps, lr,
        labeled_loader.batch_size * grad_accum,
        (pseudo_ratio * (pseudo_loader.batch_size or 0) * grad_accum) if use_pseudo else 0,
        detector.encoder_source, detector.frozen_layers, sum(p.numel() for p in trainable) / 1e6,
    )
    log.info(
        "loss: sup=%s tau=%.2f class_weights=%s | pseudo=%s norm=%s lambda %.2f->%.2f",
        loss_kwargs["loss_type"], loss_kwargs["logit_adjust_tau"],
        [round(float(w), 3) for w in class_weights] if class_weights is not None else None,
        pseudo_loss_type, pseudo_weight_norm, detector.lambda_init, detector.lambda_final,
    )

    labeled_iter = _infinite_cycle(labeled_loader)
    pseudo_iter = _infinite_cycle(pseudo_loader) if use_pseudo else None
    best: dict = {"step": 0, "val_metrics": None, "state": None}
    best_f1 = -1.0
    history: list[dict] = []
    skipped_nonfinite = 0
    win = {"loss": 0.0, "sup": 0.0, "pseudo": 0.0, "n": 0}
    t_window = time.time()

    pbar = tqdm(range(1, total_steps + 1), desc="train", unit="step", leave=False, dynamic_ncols=True)
    for step in pbar:
        detector.train()
        lam = detector.get_lambda(step - 1, total_steps)
        for _ in range(grad_accum):
            labeled_batch = next(labeled_iter)
            input_ids, attention_mask, token_type_ids = _to_device(labeled_batch, device)
            labels = labeled_batch["label"].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda"), dtype=amp_dtype):
                labeled_logits = detector.forward_reasoning(input_ids, attention_mask, token_type_ids)
                if use_pseudo:
                    pseudo = _concat_batches([next(pseudo_iter) for _ in range(pseudo_ratio)], pad_token_id)
                    p_ids = pseudo["input_ids"].to(device)
                    p_mask = pseudo["attention_mask"].to(device)
                    p_tt = pseudo["token_type_ids"].to(device)
                    pseudo_logits = detector.forward_reasoning(p_ids, p_mask, p_tt)
                    loss_dict = detector.compute_joint_loss(
                        labeled_logits=labeled_logits,
                        labeled_labels=labels,
                        pseudo_logits=pseudo_logits,
                        pseudo_labels=pseudo["label"].to(device),
                        pseudo_weights=pseudo["weight"].to(device).float(),
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
                log.warning("Non-finite loss at step=%d (skipped=%d)", step, skipped_nonfinite)
                continue
            loss = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            win["loss"] += float(loss_dict["total"].item())
            win["sup"] += float(loss_dict["supervised"].item())
            win["pseudo"] += float(loss_dict["pseudo"].item())
            win["n"] += 1

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0 and win["n"]:
            pbar.set_postfix(loss=f"{win['loss'] / win['n']:.4f}", lam=f"{lam:.3f}")

        if step in evals:
            n = max(win["n"], 1)
            row = {
                "step": step,
                "train_loss": win["loss"] / n,
                "train_sup_loss": win["sup"] / n,
                "train_pseudo_loss": (win["pseudo"] / n) if use_pseudo else 0.0,
                "lambda": lam,
                "lr": float(scheduler.get_last_lr()[0]),
                "window_seconds": round(time.time() - t_window, 1),
            }
            val_metrics = evaluate(detector, val_loader, device, f"val@{step}", use_amp, amp_dtype)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(row)
            log.info(
                "[step %d/%d] loss=%.4f sup=%.4f pseudo=%.4f lam=%.3f lr=%.2e | val acc=%.4f macro-F1=%.4f AUC=%.4f (%.0fs)",
                step, total_steps, row["train_loss"], row["train_sup_loss"], row["train_pseudo_loss"], lam, row["lr"],
                val_metrics["accuracy"], val_metrics["macro_f1"], val_metrics["auc"], row["window_seconds"],
            )
            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                best = {
                    "step": step,
                    "val_metrics": val_metrics,
                    "state": {k: v.detach().to("cpu", copy=True) for k, v in detector.state_dict().items()},
                }
                log.info("  New best val macro-F1=%.4f at step %d", best_f1, step)
            win = {"loss": 0.0, "sup": 0.0, "pseudo": 0.0, "n": 0}
            t_window = time.time()

    best["skipped_nonfinite"] = skipped_nonfinite
    log.info("Training complete. Best val macro-F1: %.4f at step %d/%d", best_f1, best["step"], total_steps)
    return history, best


def save_detector_plots(history: list[dict], outputs_dir: Path, test_metrics: Optional[dict] = None,
                        best_step: Optional[int] = None) -> None:
    if not history:
        return
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.plot(steps, [h["train_sup_loss"] for h in history], marker="o", label="supervised")
    ax1.plot(steps, [h["train_pseudo_loss"] for h in history], marker="s", label="pseudo")
    ax1.set_title("Detector train loss (mean over each eval window)")
    ax1.legend()
    for ax, key, title in ((ax2, "accuracy", "Accuracy"), (ax3, "macro_f1", "Macro-F1"), (ax4, "auc", "AUC (OvR)")):
        ax.plot(steps, [h.get(f"val_{key}", float("nan")) for h in history], marker="o", label=f"val {key}")
        if test_metrics is not None and not np.isnan(float(test_metrics.get(key, float("nan")))):
            ax.axhline(float(test_metrics[key]), linestyle=":", color="black", label=f"test {key}")
        if best_step:
            ax.axvline(best_step, linestyle="--", color="tab:red", alpha=0.5, label="best step")
        ax.set_title(title)
        ax.legend()
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlabel("Optimizer step")
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outputs_dir / "detector_training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 (v3): train the detector for one ablation method.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=ALL_METHODS)
    parser.add_argument("--seed", type=int, default=None, help="Split seed of the run (bookkeeping; default training.seed).")
    parser.add_argument("--train_seed", type=int, default=None,
                        help="Seed for initialisation / dropout / data order (default: --seed).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--keep_checkpoint", action="store_true", help="Also write the best weights to checkpoints/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = RunPaths(args.run_dir)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("models", {})
    hyper_cfg = cfg.get("hyperparameters", {})
    imbalance_cfg = cfg.get("imbalance", {})
    algo_cfg = cfg.get("algorithm", {})

    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    train_seed = args.train_seed if args.train_seed is not None else seed
    tag = detector_tag(args.method, seed, train_seed)
    fingerprint = config_fingerprint(cfg)
    _set_seed(train_seed)
    device = torch.device(args.device) if args.device else _resolve_device()
    if device.type == "cuda":
        use_tf32 = bool(train_cfg.get("use_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cudnn.benchmark = True
    log.info("run=%s method=%s tag=%s seed=%d train_seed=%d device=%s config=%s",
             paths.root.name, args.method, tag, seed, train_seed, device, fingerprint)

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
    # Same tokenizer as the extractor (the NLI cross-encoder shares deberta-v3-large's vocabulary).
    tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_kwargs)

    max_length = int(train_cfg.get("max_length", 512))
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    dynamic_padding = bool(train_cfg.get("dynamic_padding", True))
    if dynamic_padding:
        loader_kwargs["collate_fn"] = DynamicPaddingCollator(tokenizer.pad_token_id)
    ds_kwargs = {"max_length": max_length, "dynamic_padding": dynamic_padding}
    train_dataset = ClaimEvidenceDataset(str(train_path), tokenizer, **ds_kwargs)
    dev_dataset = ClaimEvidenceDataset(str(dev_path), tokenizer, **ds_kwargs)
    test_dataset = ClaimEvidenceDataset(str(test_path), tokenizer, **ds_kwargs)

    # M5: drop the last partial labeled batch (436 = 27 x 16 + 4 would give a 4-pair update every pass).
    labeled_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                drop_last=len(train_dataset) >= batch_size, **loader_kwargs)
    val_loader = DataLoader(dev_dataset, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)
    _summarize_label_distribution(train_dataset.label_ids, "Labeled train")

    out_dir = paths.detector_outputs(tag)
    pseudo_loader = None
    pseudo_size = 0
    pseudo_counts = None
    if pseudo_path is not None:
        pseudo_dataset = ClaimEvidenceDataset(str(pseudo_path), tokenizer, **ds_kwargs)
        pseudo_size = len(pseudo_dataset)
        if pseudo_size == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            reason = f"pseudo set for method {args.method} is empty: {pseudo_path}"
            with open(out_dir / "skipped.json", "w", encoding="utf-8") as fh:
                json.dump({"run": paths.root.name, "method": args.method, "tag": tag, "seed": seed,
                           "train_seed": train_seed, "reason": reason}, fh, indent=2)
            log.warning("SKIPPED: %s", reason)
            return
        pseudo_counts = _summarize_label_distribution(pseudo_dataset.label_ids, f"Pseudo set {args.method}")
        sampler = None
        if bool(imbalance_cfg.get("use_balanced_pseudo_sampler", True)):
            sampler = _build_balanced_sampler_from_labels(
                pseudo_dataset.label_ids, power=float(imbalance_cfg.get("pseudo_sampler_power", 1.0))
            )
        pseudo_bs = int(train_cfg.get("pseudo_batch_size", batch_size))
        pseudo_loader = DataLoader(
            pseudo_dataset,
            batch_size=pseudo_bs,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=pseudo_size >= pseudo_bs,
            **loader_kwargs,
        )

    detector = DualChannelDetector(
        model_name=model_name,
        num_labels=NUM_LABELS,
        lambda_init=float(hyper_cfg.get("lambda_init", 0.3)),
        lambda_final=float(hyper_cfg.get("lambda_final", 0.7)),
        cache_dir=str(model_cache_dir),
        local_files_only=local_only,
        encoder_name=model_cfg.get("nli_model") if bool(train_cfg.get("detector_init_from_nli", False)) else None,
        freeze_layers=int(train_cfg.get("detector_freeze_layers", 0)),
    ).to(device).float()

    class_weights = class_weights_from_labels(
        train_dataset.label_ids,
        power=float(imbalance_cfg.get("class_weight_power", 1.0)),
        normalize=bool(imbalance_cfg.get("normalize_class_weights", True)),
    )
    class_priors = compute_class_priors(str(train_path)) if float(algo_cfg.get("logit_adjust_tau", 0.0)) > 0 else None

    t0 = time.time()
    history, best = train(
        detector=detector,
        labeled_loader=labeled_loader,
        pseudo_loader=pseudo_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        class_weights=class_weights,
        class_priors=class_priors,
        pad_token_id=tokenizer.pad_token_id,
    )
    train_minutes = (time.time() - t0) / 60

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)

    if best["state"] is None:
        log.error("No evaluation was run; nothing to test.")
        sys.exit(1)
    detector.load_state_dict(best["state"])
    if args.keep_checkpoint:
        ckpt_path = paths.detector_ckpt(tag)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": best["step"], "model_state_dict": best["state"], "val_metrics": best["val_metrics"]}, ckpt_path)
        log.info("Saved best weights -> %s", ckpt_path)
    del best["state"]

    use_bf16 = bool(train_cfg.get("use_bf16", False))
    test_metrics, predictions = evaluate(
        detector, test_loader, device, "test",
        use_amp=bool(use_bf16 or train_cfg.get("use_fp16", False)),
        amp_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        return_predictions=True,
    )

    total_steps, eval_every = training_budget(train_cfg)
    results = {
        "run": paths.root.name,
        "method": args.method,
        "tag": tag,
        "seed": seed,
        "train_seed": train_seed,
        "config_fingerprint": fingerprint,
        "encoder_source": detector.encoder_source,
        "frozen_layers": detector.frozen_layers,
        "total_optimizer_steps": history[-1]["step"] if history else 0,
        "planned_steps": total_steps,
        "eval_every": eval_every,
        "best_step": best["step"],
        "best_val": best["val_metrics"],
        "skipped_nonfinite": best.get("skipped_nonfinite", 0),
        "pseudo_set": str(pseudo_path) if pseudo_path else None,
        "pseudo_size": pseudo_size,
        "pseudo_label_counts": pseudo_counts,
        "labeled_size": len(train_dataset),
        "test_gold_counts": _label_counts(p["gold"] for p in predictions),
        "test_pred_counts": _label_counts(p["pred"] for p in predictions),
        "train_minutes": round(train_minutes, 1),
        **test_metrics,
    }
    with open(out_dir / "test_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    with open(out_dir / "test_predictions.jsonl", "w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        save_detector_plots(history, out_dir, test_metrics, best_step=results["best_step"])
    except Exception as e:  # plotting must never fail a run
        log.warning("Plot generation skipped due to error: %s", e)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
