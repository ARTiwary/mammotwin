"""
Probability calibration diagnostics: reliability diagram and Brier score.
Called for explicitly in Phase 9 (calibrate predicted probabilities) and
again in Phase 14's evaluation criteria — built once here, reusable by any
classifier's evaluation step.

A well-calibrated model's predicted probability should match its actual
accuracy: among all predictions where the model said "70% malignant," about
70% should actually be malignant. The reliability diagram plots this
directly; the Brier score summarizes it as a single number (mean squared
error between predicted probability and the true 0/1 outcome — lower is
better, 0 is perfect).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def compute_calibration_metrics(y_true, y_prob, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    brier = brier_score_loss(y_true, y_prob)

    if len(np.unique(y_true)) < 2:
        return {"brier_score": brier, "fraction_positives": None, "mean_predicted_value": None}

    fraction_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return {
        "brier_score": brier,
        "fraction_positives": fraction_positives,
        "mean_predicted_value": mean_predicted_value,
    }


def plot_reliability_diagram(y_true, y_prob, out_path: str, n_bins: int = 10, title: str = "Reliability Diagram"):
    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    if metrics["fraction_positives"] is not None:
        ax.plot(metrics["mean_predicted_value"], metrics["fraction_positives"],
                marker="o", label="Model")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual)")
    ax.set_title(f"{title}\nBrier score: {metrics['brier_score']:.4f}")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    return metrics