"""
Shared inverse-frequency class weighting, used by any classifier training
script (Phase 6's whole-image baseline, Phase 9's lesion-crop classifier,
and any future variant) so the majority class doesn't dominate the loss on
an imbalanced dataset.
"""

import torch

LABEL_MAP = {"benign": 0, "malignant": 1}


def compute_class_weights(class_counts: dict, num_classes: int) -> torch.Tensor:
    """class_counts: e.g. {'benign': 1470, 'malignant': 1031} (from a
    dataset's .class_counts() method)."""
    total = sum(class_counts.values())
    weights = torch.ones(num_classes)
    for label_name, label_idx in LABEL_MAP.items():
        count = class_counts.get(label_name, 0)
        if count > 0:
            weights[label_idx] = total / (num_classes * count)
    return weights