"""
Phase 12: simulate a longitudinal series from a single REAL mammogram +
its real Phase 7 lesion mask.

CBIS-DDSM has no genuine prior/current exam pairs for the same patient
(confirmed in the Phase 1 project plan) — per the plan's own guidance,
the longitudinal module uses SIMULATED pairs, clearly labeled as such,
purely to demonstrate the software's capability.

Approach: take one real "current" exam and mask, then synthesize earlier
"visits" by shrinking/removing the lesion (via inpainting real surrounding
tissue texture — the anatomy stays real, only the lesion's presence/size
across time is synthetic), and applying a small random misalignment (real
mammography visits are never pixel-perfectly aligned) that the
registration step must then correct. This makes the pipeline do genuine
alignment and change-detection work, not just compare identical images.
"""

import cv2
import numpy as np


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img_min, img_max = img.min(), img.max()
    denom = (img_max - img_min) if (img_max - img_min) > 1e-6 else 1.0
    return ((img - img_min) / denom * 255.0).astype(np.uint8)


def shrink_lesion(image: np.ndarray, mask: np.ndarray, area_fraction: float):
    """
    Returns (modified_image, modified_mask) where the lesion has been
    shrunk to approximately area_fraction of its original size (0 = fully
    removed, 1 = unchanged), using a distance-transform-based shrink (keeps
    the most "interior" part of the lesion shape) and inpainting to blend
    the removed portion using real surrounding tissue texture.
    """
    image_uint8 = _to_uint8(image)
    mask_binary = (mask > 127).astype(np.uint8) * 255

    if area_fraction >= 0.999:
        return image_uint8.copy(), mask_binary.copy()

    if mask_binary.sum() == 0:
        return image_uint8.copy(), mask_binary.copy()

    if area_fraction <= 0.001:
        shrunk_mask = np.zeros_like(mask_binary)
    else:
        dist = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        inside_values = dist[mask_binary > 0]
        threshold = np.quantile(inside_values, 1 - area_fraction)
        shrunk_mask = ((dist >= threshold) & (mask_binary > 0)).astype(np.uint8) * 255

    removal_mask = cv2.subtract(mask_binary, shrunk_mask)
    inpainted = cv2.inpaint(image_uint8, removal_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return inpainted, shrunk_mask


def apply_random_misalignment(image: np.ndarray, seed: int,
                               max_translation: float = 15.0, max_rotation_deg: float = 3.0):
    """Simulates imperfect patient positioning between real mammography
    visits — a small random rotation + translation. Returns the shifted
    image and the exact transform matrix used (so registration accuracy
    can later be checked against this known ground truth)."""
    rng = np.random.default_rng(seed)
    tx, ty = rng.uniform(-max_translation, max_translation, size=2)
    angle = rng.uniform(-max_rotation_deg, max_rotation_deg)

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty

    transformed = cv2.warpAffine(_to_uint8(image), M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return transformed, M


def simulate_timeline(image: np.ndarray, mask: np.ndarray, seed: int = 42,
                       add_misalignment: bool = True):
    """
    Builds a 3-point SIMULATED timeline from one real exam:
      baseline:  lesion fully absent (area_fraction=0)
      follow_up: lesion at ~50% of its final size
      current:   the real, unmodified exam (area_fraction=1)

    Each earlier timepoint additionally gets a small random misalignment,
    since real mammography visits are never pixel-perfectly registered to
    each other — the registration step is meant to correct exactly this.
    """
    rng = np.random.default_rng(seed)

    baseline_img, baseline_mask = shrink_lesion(image, mask, area_fraction=0.0)
    followup_img, followup_mask = shrink_lesion(image, mask, area_fraction=0.5)
    current_img, current_mask = _to_uint8(image), (mask > 127).astype(np.uint8) * 255

    transforms = {}
    if add_misalignment:
        baseline_img, M_baseline = apply_random_misalignment(baseline_img, seed=int(rng.integers(0, 1_000_000)))
        baseline_mask, _ = apply_random_misalignment(baseline_mask, seed=int(rng.integers(0, 1_000_000)))
        followup_img, M_followup = apply_random_misalignment(followup_img, seed=int(rng.integers(0, 1_000_000)))
        followup_mask, _ = apply_random_misalignment(followup_mask, seed=int(rng.integers(0, 1_000_000)))
        transforms = {"baseline": M_baseline, "followup": M_followup}

    return {
        "baseline": {"image": baseline_img, "mask": baseline_mask},
        "followup": {"image": followup_img, "mask": followup_mask},
        "current": {"image": current_img, "mask": current_mask},
        "applied_transforms": transforms,
    }