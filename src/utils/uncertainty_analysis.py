"""
Phase 11: analysis tools for turning per-example uncertainty scores into an
actionable, DATA-DRIVEN review threshold, and for checking whether
uncertainty actually predicts errors (rather than just assuming it does).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def error_detection_auroc(uncertainty_scores, is_incorrect) -> float:
    """
    A standard sanity check in the uncertainty literature: if uncertainty
    is doing its job, examples the model got WRONG should generally have
    HIGHER uncertainty scores than examples it got right. Treating
    'is_incorrect' as the target and the uncertainty score as the
    prediction, ROC-AUC measures exactly this — 0.5 means uncertainty is
    no better than random at flagging errors; higher is better.
    """
    if len(np.unique(is_incorrect)) < 2:
        return None  # can't compute AUROC with only one class present (e.g. zero errors)
    return roc_auc_score(is_incorrect, uncertainty_scores)


def review_threshold_sweep(confidence_scores, is_incorrect, thresholds=None) -> pd.DataFrame:
    """
    For each candidate confidence threshold, reports what fraction of
    examples would be flagged for review (confidence < threshold), and the
    actual error rate inside vs. outside that flagged group. This is what
    "set a review threshold using validation data" means in practice: you
    can look at this table and pick the threshold that catches an
    acceptable fraction of errors at an acceptable review workload.
    """
    if thresholds is None:
        thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    confidence_scores = np.asarray(confidence_scores)
    is_incorrect = np.asarray(is_incorrect)
    n_total = len(confidence_scores)

    rows = []
    for t in thresholds:
        flagged = confidence_scores < t
        n_flagged = int(flagged.sum())
        pct_flagged = n_flagged / n_total if n_total > 0 else 0.0

        error_rate_flagged = is_incorrect[flagged].mean() if n_flagged > 0 else None
        error_rate_unflagged = is_incorrect[~flagged].mean() if (n_total - n_flagged) > 0 else None

        # Of all actual errors, what fraction did this threshold catch?
        total_errors = is_incorrect.sum()
        errors_caught = is_incorrect[flagged].sum() if n_flagged > 0 else 0
        error_recall = errors_caught / total_errors if total_errors > 0 else None

        rows.append({
            "threshold": t,
            "pct_flagged": pct_flagged,
            "error_rate_in_flagged": error_rate_flagged,
            "error_rate_in_unflagged": error_rate_unflagged,
            "fraction_of_all_errors_caught": error_recall,
        })

    return pd.DataFrame(rows)