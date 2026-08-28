"""
Phase 12: quantify change between two (already-registered) timepoints —
lesion-area change, spatial overlap, and, where a classifier is provided,
change in model score. These are the research features the plan calls for:
"lesion-area change, region overlap and change in model score."
"""

import numpy as np


def lesion_area_change(mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    """mask_a = earlier timepoint, mask_b = later timepoint, both already
    in the SAME (registered) coordinate frame."""
    area_a = float((mask_a > 127).sum())
    area_b = float((mask_b > 127).sum())
    denom = max(area_a, area_b, 1.0)
    return {
        "area_a": area_a, "area_b": area_b,
        "absolute_change": area_b - area_a,
        "relative_change": (area_b - area_a) / denom,
    }


def region_overlap_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = (mask_a > 127)
    b = (mask_b > 127)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection) / float(union) if union > 0 else (1.0 if intersection == 0 else 0.0)


def pixel_difference_map(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Simple absolute-difference change map between two REGISTERED images
    — highlights newly appearing / changed regions for visualization."""
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32)
    return np.abs(b - a)


def compute_change_features(mask_a: np.ndarray, mask_b: np.ndarray,
                             image_a: np.ndarray, image_b: np.ndarray,
                             model_score_a: float = None, model_score_b: float = None) -> dict:
    features = lesion_area_change(mask_a, mask_b)
    features["region_overlap_iou"] = region_overlap_iou(mask_a, mask_b)
    features["diff_map"] = pixel_difference_map(image_a, image_b)

    if model_score_a is not None and model_score_b is not None:
        features["model_score_a"] = model_score_a
        features["model_score_b"] = model_score_b
        features["model_score_change"] = model_score_b - model_score_a

    return features