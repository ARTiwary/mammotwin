"""
Reusable classification metrics. Used from Phase 6 onward so every module
(baseline classifier, lesion-crop classifier, multimodal fusion) reports
results the exact same way, per Phase 14's fixed metric list:
ROC-AUC, PR-AUC, sensitivity, specificity, precision, F1, balanced accuracy.
"""

import numpy as np
from src.utils.stats import (
    roc_auc_score, average_precision_score, confusion_matrix_binary,
    precision_score, f1_score, balanced_accuracy_score, accuracy_score,
)


def compute_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """
    y_true: array of 0/1 ground-truth labels
    y_prob: array of predicted probability of the POSITIVE class (malignant=1)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix_binary(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # a.k.a. recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # ROC-AUC / PR-AUC need both classes present to be defined.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def format_metrics_report(metrics: dict) -> str:
    lines = ["=== Classification Metrics ==="]
    for key in ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
                "precision", "f1", "roc_auc", "pr_auc"]:
        val = metrics.get(key)
        lines.append(f"  {key:20s}: {val:.4f}" if val is not None else f"  {key:20s}: N/A")
    cm = metrics["confusion_matrix"]
    lines.append(f"  Confusion matrix    : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    return "\n".join(lines)