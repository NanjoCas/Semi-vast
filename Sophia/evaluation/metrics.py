"""
evaluation/metrics.py
=====================
Evaluation metrics for the fake news detection (claim verification) pipeline.

Supported metrics:
    - Accuracy
    - Macro-F1
    - Per-class precision, recall, F1
    - AUC (one-vs-rest, requires probability scores)
    - Confusion matrix
    - Logic consistency score (mean |logic_score| on correctly classified samples)

Label schema:
    SUPPORTS       -> 0
    REFUTES        -> 1
    NOT_ENOUGH_INFO -> 2
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

LABEL2ID: dict[str, int] = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}
ID2LABEL: dict[int, str] = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT_ENOUGH_INFO"}

_NUM_CLASSES = len(LABEL2ID)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_probs: list[list[float]],
) -> dict:
    """
    Compute a comprehensive set of evaluation metrics for 3-way claim
    verification.

    Args:
        y_true:  Ground-truth integer class labels, one per sample.
        y_pred:  Predicted integer class labels, one per sample.
        y_probs: Per-sample class probabilities, shape (N, 3).  Each inner
                 list must sum to ~1.0 and have exactly 3 elements ordered
                 [P(SUPPORTS), P(REFUTES), P(NOT_ENOUGH_INFO)].

    Returns:
        dict with the following keys:
            "accuracy"              (float)
            "macro_f1"              (float)
            "per_class_precision"   (dict[str, float])  label_name -> value
            "per_class_recall"      (dict[str, float])  label_name -> value
            "per_class_f1"          (dict[str, float])  label_name -> value
            "auc_ovr"               (float)  one-vs-rest multi-class AUC
            "confusion_matrix"      (list[list[int]])  row=true, col=pred
    """
    if len(y_true) == 0:
        raise ValueError("y_true is empty; cannot compute metrics.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)}."
        )
    if len(y_true) != len(y_probs):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_probs has {len(y_probs)}."
        )

    # --- Accuracy -----------------------------------------------------------
    accuracy = float(accuracy_score(y_true, y_pred))

    # --- Macro F1 -----------------------------------------------------------
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    # --- Per-class precision / recall / F1 ----------------------------------
    labels = list(range(_NUM_CLASSES))

    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    per_class_precision: dict[str, float] = {
        ID2LABEL[i]: float(precision_arr[i]) for i in labels
    }
    per_class_recall: dict[str, float] = {
        ID2LABEL[i]: float(recall_arr[i]) for i in labels
    }
    per_class_f1: dict[str, float] = {
        ID2LABEL[i]: float(f1_arr[i]) for i in labels
    }

    # --- AUC (one-vs-rest) --------------------------------------------------
    auc_ovr: float | None = None
    try:
        # roc_auc_score expects shape (N, num_classes) for multi_class="ovr"
        auc_ovr = float(
            roc_auc_score(
                y_true,
                y_probs,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
        )
    except Exception as exc:  # noqa: BLE001
        # Can fail if a class is absent from y_true in very small splits
        auc_ovr = None
        import warnings
        warnings.warn(
            f"[compute_metrics] AUC computation failed: {exc}. "
            "Setting auc_ovr=None.",
            RuntimeWarning,
            stacklevel=2,
        )

    # --- Confusion matrix ---------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "auc_ovr": auc_ovr,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Logic consistency score
# ---------------------------------------------------------------------------

def compute_logic_consistency_score(predictions: list[dict]) -> float:
    """
    Compute the mean absolute logic score over *correctly classified* samples.

    This measures how logically consistent the model's correct predictions are:
    a higher value means that on the samples the model got right, the evidence
    strongly entailed (or contradicted) the claim, rather than being neutral.

    Args:
        predictions: List of dicts, each containing:
            "predicted_label" (int): Model's predicted class id.
            "true_label"      (int): Ground-truth class id.
            "logic_score"     (float): LogicScore in [-1, 1] from LogicScorer.

    Returns:
        float: Mean |logic_score| of correctly classified samples, in [0, 1].
               Returns 0.0 if there are no correctly classified samples.
    """
    if not predictions:
        return 0.0

    correct_logic_scores: list[float] = [
        abs(float(p["logic_score"]))
        for p in predictions
        if p.get("predicted_label") == p.get("true_label")
    ]

    if not correct_logic_scores:
        return 0.0

    return sum(correct_logic_scores) / len(correct_logic_scores)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_metrics_report(metrics: dict, split_name: str = "test") -> str:
    """
    Format a metrics dict into a human-readable report string.

    Args:
        metrics:    Output dict from :func:`compute_metrics`, optionally
                    augmented with a ``"logic_consistency"`` key (float).
        split_name: Name of the evaluation split (e.g. "test",
                    "climatemist_weak") printed in the header.

    Returns:
        str: Multi-line formatted report, suitable for printing or file I/O.
    """
    sep_wide  = "=" * 62
    sep_thin  = "-" * 62

    lines: list[str] = [
        "",
        sep_wide,
        f"  Evaluation Report  —  split: {split_name}",
        sep_wide,
        "",
    ]

    # --- Top-level scalars --------------------------------------------------
    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("macro_f1")
    auc_ovr  = metrics.get("auc_ovr")
    logic_cs = metrics.get("logic_consistency")

    if accuracy is not None:
        lines.append(f"  Accuracy          : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    if macro_f1 is not None:
        lines.append(f"  Macro F1          : {macro_f1:.4f}")
    if auc_ovr is not None:
        lines.append(f"  AUC (OvR macro)   : {auc_ovr:.4f}")
    else:
        lines.append("  AUC (OvR macro)   : N/A")
    if logic_cs is not None:
        lines.append(f"  Logic Consistency : {logic_cs:.4f}")

    lines.append("")
    lines.append(sep_thin)

    # --- Per-class table ----------------------------------------------------
    per_prec = metrics.get("per_class_precision", {})
    per_rec  = metrics.get("per_class_recall", {})
    per_f1   = metrics.get("per_class_f1", {})

    if per_prec or per_rec or per_f1:
        col_w = 18
        header = (
            f"  {'Class':<{col_w}}"
            f"{'Precision':>10}"
            f"{'Recall':>10}"
            f"{'F1':>10}"
        )
        lines.append(header)
        lines.append(sep_thin)

        for label_name in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"):
            prec = per_prec.get(label_name, float("nan"))
            rec  = per_rec.get(label_name, float("nan"))
            f1   = per_f1.get(label_name, float("nan"))
            lines.append(
                f"  {label_name:<{col_w}}"
                f"{prec:>10.4f}"
                f"{rec:>10.4f}"
                f"{f1:>10.4f}"
            )

    # --- Confusion matrix ---------------------------------------------------
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        lines.append("")
        lines.append(sep_thin)
        lines.append("  Confusion Matrix  (rows=true, cols=pred)")
        lines.append("")

        label_names = ["SUPPORTS", "REFUTES", "NEI"]
        col_w_cm = 14

        # Header row
        header_row = f"  {'':>{col_w_cm}}"
        for lbl in label_names:
            header_row += f"  {lbl:>{col_w_cm}}"
        lines.append(header_row)

        for i, row in enumerate(cm):
            row_label = label_names[i] if i < len(label_names) else str(i)
            row_str = f"  {row_label:>{col_w_cm}}"
            for val in row:
                row_str += f"  {val:>{col_w_cm}}"
            lines.append(row_str)

    lines.append("")
    lines.append(sep_wide)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baseline comparison table
# ---------------------------------------------------------------------------

def compare_baselines(results_dict: dict) -> str:
    """
    Produce a formatted comparison table across multiple model configurations.

    Args:
        results_dict: Mapping of run_name -> metrics_dict, e.g.:
            {
                "baseline_a":  metrics_dict_a,
                "baseline_b":  metrics_dict_b,
                "full_model":  metrics_dict_full,
            }
            Each metrics_dict is the output of :func:`compute_metrics`,
            optionally augmented with ``"logic_consistency"`` (float).

    Returns:
        str: Formatted comparison table string.
    """
    if not results_dict:
        return "  [compare_baselines] No results provided.\n"

    sep_wide = "=" * 86
    sep_thin = "-" * 86

    lines: list[str] = [
        "",
        sep_wide,
        "  Baseline Comparison Table",
        sep_wide,
        "",
    ]

    # Column header
    col_name = 22
    col_val  = 12

    header = (
        f"  {'Model':<{col_name}}"
        f"{'Accuracy':>{col_val}}"
        f"{'Macro F1':>{col_val}}"
        f"{'AUC (OvR)':>{col_val}}"
        f"{'Logic Cons.':>{col_val}}"
    )
    lines.append(header)
    lines.append(sep_thin)

    # One row per model
    for run_name, metrics in results_dict.items():
        if metrics is None:
            lines.append(
                f"  {run_name:<{col_name}}"
                + f"{'N/A':>{col_val}}" * 4
            )
            continue

        accuracy  = metrics.get("accuracy")
        macro_f1  = metrics.get("macro_f1")
        auc_ovr   = metrics.get("auc_ovr")
        logic_cs  = metrics.get("logic_consistency")

        def _fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "N/A"

        lines.append(
            f"  {run_name:<{col_name}}"
            f"{_fmt(accuracy):>{col_val}}"
            f"{_fmt(macro_f1):>{col_val}}"
            f"{_fmt(auc_ovr):>{col_val}}"
            f"{_fmt(logic_cs):>{col_val}}"
        )

    lines.append(sep_thin)

    # Per-class F1 sub-table
    lines.append("")
    lines.append("  Per-Class F1 Breakdown")
    lines.append(sep_thin)

    col_val_wide = 14

    sub_header = (
        f"  {'Model':<{col_name}}"
        f"{'SUPPORTS':>{col_val_wide}}"
        f"{'REFUTES':>{col_val_wide}}"
        f"{'NOT_ENOUGH_INFO':>{col_val_wide}}"
    )
    lines.append(sub_header)
    lines.append(sep_thin)

    for run_name, metrics in results_dict.items():
        if metrics is None:
            lines.append(
                f"  {run_name:<{col_name}}"
                + f"{'N/A':>{col_val_wide}}" * 3
            )
            continue

        per_f1 = metrics.get("per_class_f1", {})
        sup_f1 = per_f1.get("SUPPORTS")
        ref_f1 = per_f1.get("REFUTES")
        nei_f1 = per_f1.get("NOT_ENOUGH_INFO")

        def _fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "N/A"

        lines.append(
            f"  {run_name:<{col_name}}"
            f"{_fmt(sup_f1):>{col_val_wide}}"
            f"{_fmt(ref_f1):>{col_val_wide}}"
            f"{_fmt(nei_f1):>{col_val_wide}}"
        )

    lines.append("")
    lines.append(sep_wide)
    lines.append("")

    return "\n".join(lines)
