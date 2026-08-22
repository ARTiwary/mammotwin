"""
Image loading utilities for MammoTwin.

Handles two input types:
  1. Standard image formats (.png, .jpg, .tif) via OpenCV.
  2. DICOM files (.dcm) via pydicom, with correct handling of
     PhotometricInterpretation (MONOCHROME1 vs MONOCHROME2) — mammogram
     DICOMs are frequently MONOCHROME1, where pixel intensity is *inverted*
     relative to what you'd naively expect (higher stored value = darker).
     Getting this wrong silently corrupts every downstream step.
"""

import os
import numpy as np
import cv2

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


def is_dicom(path: str) -> bool:
    return path.lower().endswith((".dcm", ".dicom"))


def load_dicom(path: str) -> np.ndarray:
    """Load a DICOM file and return a float32 grayscale array, correctly
    oriented (i.e. after undoing MONOCHROME1 inversion if present)."""
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required to load .dcm files. "
                           "Install with: pip install pydicom")

    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present (raw -> real-world units)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    # MONOCHROME1 means stored pixel values are inverted (0 = white/bright).
    # Flip it so that, like MONOCHROME2, higher value = brighter, consistently.
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        img = img.max() - img

    return img


def load_standard_image(path: str) -> np.ndarray:
    """Load a standard image format as float32 grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img.astype(np.float32)


def load_image(path: str) -> np.ndarray:
    """Auto-detect format and load as a float32 grayscale array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at: {path}")

    if is_dicom(path):
        return load_dicom(path)
    return load_standard_image(path)


def generate_synthetic_mammogram(size: int = 512, seed: int = 42) -> np.ndarray:
    """
    Generate a synthetic grayscale test image that stands in for a mammogram
    during Phase 2, before real data has been downloaded (Phase 3).
    This is ONLY for verifying the load -> preprocess -> display pipeline
    mechanics — it has no diagnostic meaning and must never be used for
    training or evaluation.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(loc=60, scale=8, size=(size, size)).astype(np.float32)

    # Soft tissue-like background gradient
    yy, xx = np.mgrid[0:size, 0:size]
    breast_mask = ((xx - size * 0.35) ** 2 / (size * 0.4) ** 2 +
                    (yy - size * 0.5) ** 2 / (size * 0.45) ** 2) < 1
    img[breast_mask] += 60

    # A brighter round blob standing in for a "lesion"
    cy, cx, r = int(size * 0.45), int(size * 0.4), size // 18
    lesion_mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
    img[lesion_mask] += 90

    img = np.clip(img, 0, 255)
    return img