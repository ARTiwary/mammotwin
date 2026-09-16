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


def plot_confusion_matrix(cm: dict, path: str, title: str = "Confusion Matrix",
                           class_names=("benign", "malignant")):
    """
    Saves a 2x2 confusion matrix heatmap with both raw counts and
    row-normalized percentages annotated in each cell (e.g. "160\\n61.3%
    of malignant") -- the percentage is what actually answers "of the
    real malignant cases, what fraction did we catch," which the raw
    count alone doesn't make obvious at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1  # guard divide-by-zero on a degenerate empty class
    pct = matrix / row_totals * 100

    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Predicted\n{class_names[0]}", f"Predicted\n{class_names[1]}"])
    ax.set_yticklabels([f"Actual\n{class_names[0]}", f"Actual\n{class_names[1]}"])

    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, f"{matrix[i, j]}\n({pct[i, j]:.1f}%)",
                    ha="center", va="center", color=color, fontsize=12)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path