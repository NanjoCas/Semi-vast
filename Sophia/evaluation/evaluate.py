"""
evaluation/evaluate.py
----------------------
Run end-to-end evaluation for the dual-channel detector.

Usage:
    python evaluation/evaluate.py --config configs/config.yaml --split test
    python evaluation/evaluate.py --config configs/config.yaml --split climatemist_weak
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_pipeline import ClaimEvidenceDataset  # noqa: E402
from models.detector import DualChannelDetector  # noqa: E402
from evaluation.metrics import compute_metrics, format_metrics_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the dual-channel detector.")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/config.yaml)")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "climatemist_weak"],
        help="Which split to evaluate (see processed/ directory).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override detector checkpoint path (defaults to checkpoints/detector/best_model.pt)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--device", default=None, help="Optional device override (cuda/cpu/mps)")
    return parser.parse_args()


def resolve_device(cli_device: str | None) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_detector(model_name: str, checkpoint_path: Path, device: torch.device) -> DualChannelDetector:
    detector = DualChannelDetector(model_name=model_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Detector checkpoint not found: {checkpoint_path}\n"
            "Run training/train_detector.py to produce it."
        )
    state_dict = torch.load(checkpoint_path, map_location=device)
    detector.load_state_dict(state_dict)
    detector.to(device)
    detector.eval()
    return detector


def evaluate_split(detector: DualChannelDetector, dataset_path: Path, tokenizer, batch_size: int, device: torch.device):
    ds = ClaimEvidenceDataset(str(dataset_path), tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_probs: list[list[float]] = []
    all_preds: list[int] = []
    all_labels: list[int] = []

    detector.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = detector.forward_reasoning(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return metrics


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    paths_cfg = config["paths"]
    models_cfg = config["models"]

    # Resolve dataset path
    split_to_path = {
        "test": PROJECT_ROOT / paths_cfg["labeled_test"],
        "climatemist_weak": PROJECT_ROOT / "processed" / "unlabeled" / "climatemist_weak_labeled.jsonl",
    }
    dataset_path = split_to_path[args.split]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation split not found: {dataset_path}\nRun run_pipeline.py first.")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else PROJECT_ROOT / "checkpoints" / "detector" / "best_model.pt"

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(models_cfg["deberta_base"])
    detector = load_detector(models_cfg["deberta_base"], checkpoint_path, device)

    metrics = evaluate_split(detector, dataset_path, tokenizer, args.batch_size, device)
    report = format_metrics_report(metrics, split_name=args.split)
    print(report)

    # Save metrics alongside checkpoint
    out_dir = checkpoint_path.parent
    out_path = out_dir / f"metrics_{args.split}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
