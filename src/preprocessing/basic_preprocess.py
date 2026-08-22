"""
Phase 2/4 preprocessing primitives.

These are intentionally simple for Phase 2 (just enough to prove the
pipeline works end-to-end on one image). Phase 4 will extend this with
proper background removal, quality checks, and Albumentations-based
augmentation — but the *interface* (preprocess_image(img, config)) should
stay stable so inference always matches training.
"""

import numpy as np
import cv2


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


def preprocess_image(img: np.ndarray, config: dict) -> np.ndarray:
    """Full preprocessing pipeline driven entirely by config, so training
    and inference are guaranteed to apply identical steps."""
    clip_percentile = tuple(config["preprocessing"]["clip_percentile"])
    image_size = tuple(config["preprocessing"]["image_size"])

    img = normalize_image(img, clip_percentile)
    img = resize_image(img, image_size)
    return img