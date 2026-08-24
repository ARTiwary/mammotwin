"""
Phase 4: quality checks to flag images unsuitable for reliable automated
analysis, per the project plan's Quality Gate stage.

Each check is cheap and runs on the raw loaded image (before background
removal/normalization), except `breast_area_fraction` which requires the
background-removal step to have already run.
"""

import numpy as np


def is_blank(img: np.ndarray, std_threshold: float = 3.0) -> bool:
    """A near-uniform image (all one color) has almost no pixel variation.
    std_threshold is on the image's own intensity scale."""
    return float(np.std(img)) < std_threshold


def is_low_contrast(img: np.ndarray, assumed_max_range: float = 255.0,
                     range_fraction_threshold: float = 0.08) -> bool:
    """Flag images where the 1st-99th percentile intensity span is a small
    fraction of the sensor/display's available range (default assumes an
    8-bit-like 0-255 scale, true for the CBIS-DDSM JPEG mirror; pass a
    different assumed_max_range for raw DICOM pixel data)."""
    p1, p99 = np.percentile(img, [1, 99])
    return float(p99 - p1) < (assumed_max_range * range_fraction_threshold)


def breast_area_too_small(breast_area_fraction: float, min_fraction: float = 0.03) -> bool:
    """After background removal, if the detected breast region is a tiny
    sliver of the whole image, segmentation likely failed or the scan is
    mostly empty/blank."""
    return breast_area_fraction < min_fraction


def run_quality_checks(img: np.ndarray, breast_area_fraction: float = None) -> dict:
    """
    Run the full Phase 4 quality gate on one already-loaded image.
    Pass breast_area_fraction (from background_removal.remove_background)
    if available, to also catch failed-segmentation / near-empty cases.

    Returns a dict of individual flags plus an overall `passed` boolean.
    An image "fails" the gate if ANY flag is True.
    """
    flags = {
        "is_blank": is_blank(img),
        "is_low_contrast": is_low_contrast(img),
    }

    if breast_area_fraction is not None:
        flags["breast_area_too_small"] = breast_area_too_small(breast_area_fraction)

    flags["passed"] = not any(flags.values())
    return flags