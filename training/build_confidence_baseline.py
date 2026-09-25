"""
build_confidence_baseline.py
=============================
Builds the pseudo-label set for Ablation Baseline B: "semi-supervised with
confidence threshold (no RL, no LogicScore)".

Unlike the full pipeline (Phase 2 weight = beta1*confidence + beta2*|logic| +
beta3*discourse, then Phase 3 PPO selection), this baseline keeps a sample
purely because its classifier confidence clears a fixed threshold - it never
looks at LogicScore, DiscourseScore, or the RL selector. All kept samples are
written with weight=1.0 (uniform), so the detector's joint loss treats every
pseudo-labeled sample the same instead of down-weighting shaky ones.

Usage:
    python training/build_confidence_baseline.py --threshold 0.7
    python training/build_confidence_baseline.py \
        --input processed/pseudo_labels/pseudo_labeled_pool.jsonl \
        --output processed/pseudo_labels/baseline_B_confidence_filtered.jsonl \
        --threshold 0.7
"""

import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ID2LABEL = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a confidence-threshold-only pseudo-label set (Ablation Baseline B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="processed/pseudo_labels/pseudo_labeled_pool.jsonl",
        help="Full unfiltered pseudo-label pool from generate_pseudolabels.py (Phase 2 output).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="processed/pseudo_labels/baseline_B_confidence_filtered.jsonl",
        help="Where to write the confidence-filtered pseudo-label set.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum classifier confidence (p_clf) required to keep a sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if not input_path.exists():
        raise FileNotFoundError(
            f"Pseudo-label pool not found: {input_path}\n"
            "Run training/generate_pseudolabels.py first."
        )

    total = 0
    kept = 0
    label_dist: Counter = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            confidence = float(rec.get("confidence", 0.0))
            if confidence < args.threshold:
                continue

            pseudo_label = rec["pseudo_label"]
            kept_rec = {
                "id": rec.get("id", ""),
                "claim": rec["claim"],
                "pseudo_label": pseudo_label,
                "source": rec.get("source", "unknown"),
                "confidence": confidence,
                "weight": 1.0,  # uniform: confidence-threshold baseline does not weight samples
            }
            fout.write(json.dumps(kept_rec, ensure_ascii=False) + "\n")
            kept += 1
            label_dist[ID2LABEL.get(pseudo_label, str(pseudo_label))] += 1

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "threshold": args.threshold,
        "total_pool": total,
        "kept": kept,
        "keep_ratio": round(kept / total, 4) if total else 0.0,
        "label_distribution": dict(label_dist),
    }
    stats_path = output_path.parent / "baseline_B_confidence_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(f"Confidence threshold : {args.threshold}")
    print(f"Pool size            : {total}")
    print(f"Kept                 : {kept} ({stats['keep_ratio'] * 100:.1f}%)")
    print(f"Label distribution   : {dict(label_dist)}")
    print(f"Written to           : {output_path}")
    print(f"Stats saved to       : {stats_path}")


if __name__ == "__main__":
    main()
