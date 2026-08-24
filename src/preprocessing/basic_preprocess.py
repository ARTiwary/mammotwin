"""
Phase 2/4 preprocessing pipeline.

Full pipeline, in order:
  1. (optional, config-driven) Background removal — crop to breast tissue,
     discard black background and label artifacts.
  2. Intensity normalization — percentile clip + min-max to [0, 1].
  3. Resize to the model's fixed input size.

The interface (preprocess_image(img, config)) is stable from Phase 2 onward
so inference always applies IDENTICAL steps to training — the single
biggest source of silent train/inference mismatch bugs.
"""

import json
import os
import numpy as np
import cv2

from src.preprocessing.background_removal import remove_background
from src.preprocessing.quality_checks import run_quality_checks


def normalize_image(img: np.ndarray, clip_percentile=(1, 99)) -> np.ndarray:
    """Clip outlier intensities, then min-max normalize to [0, 1].

    Percentile clipping (rather than raw min/max) prevents a handful of
    extreme pixels from washing out contrast across the whole image.
    """
    low, high = np.percentile(img, clip_percentile)
    img = np.clip(img, low, high)

    denom = (high - low) if (high - low) > 1e-6 else 1.0
    img = (img - low) / denom
    return img.astype(np.float32)


def resize_image(img: np.ndarray, size) -> np.ndarray:
    """Resize to (height, width). Uses area interpolation, which is the
    right choice when shrinking images (avoids aliasing on fine detail)."""
    h, w = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def preprocess_image(img: np.ndarray, config: dict, run_quality_gate: bool = True) -> dict:
    """
    Full preprocessing pipeline driven entirely by config, so training and
    inference are guaranteed to apply identical steps.

    Returns a dict:
      'processed':  final [0,1]-normalized, resized image
      'quality':    dict of quality-check flags (only if run_quality_gate)
      'bbox':       breast-region bounding box in the ORIGINAL image
                    (None if background removal was skipped)
    """
    pp_cfg = config["preprocessing"]
    qc_cfg = config.get("quality_checks", {})

    bbox = None
    breast_area_fraction = None
    working_img = img

    if pp_cfg.get("remove_background", True):
        bg_result = remove_background(img, padding=pp_cfg.get("background_padding", 10))
        working_img = bg_result["cropped"]
        bbox = bg_result["bbox"]
        breast_area_fraction = bg_result["breast_area_fraction"]

    quality = None
    if run_quality_gate:
        quality = run_quality_checks(img, breast_area_fraction=breast_area_fraction)
        # Override thresholds from config if provided, without duplicating
        # the check logic — re-run only if config differs from defaults.
        # (Kept simple here; see quality_checks.py to tune defaults directly.)

    clip_percentile = tuple(pp_cfg["clip_percentile"])
    image_size = tuple(pp_cfg["image_size"])

    normalized = normalize_image(working_img, clip_percentile)
    processed = resize_image(normalized, image_size)

    return {
        "processed": processed,
        "quality": quality,
        "bbox": bbox,
        "breast_area_fraction": breast_area_fraction,
    }


def save_preprocessing_config(config: dict, output_path: str) -> None:
    """
    Persist the EXACT preprocessing settings used for a given run of
    processed data, so it's always possible to verify later that a set of
    processed images and the model trained on them used matching settings.
    """
    record = {
        "preprocessing": config["preprocessing"],
        "quality_checks": config.get("quality_checks", {}),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)