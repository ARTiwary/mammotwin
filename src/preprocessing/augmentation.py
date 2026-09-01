"""
Data augmentation for training only (never applied to val/test), using
only numpy and cv2 — deliberately not Albumentations, which pulls in scipy
transitively (the same blocked-DLL risk this project has already hit twice
via sklearn/torchmetrics).

Applied to the RAW image (before Phase 4's normalize/resize), so it stacks
naturally with the existing preprocessing pipeline. For paired data
(image+mask, image+bbox), the SAME random parameters must be applied to
both — use augment_pair() for that, not two independent calls to
augment_image(), which would desynchronize them.
"""

import cv2
import numpy as np


def _random_flip_params(rng):
    return {"flip": rng.random() < 0.5}


def _random_rotation_params(rng, max_degrees=15):
    return {"angle": rng.uniform(-max_degrees, max_degrees)}


def _random_brightness_contrast_params(rng, brightness_range=0.15, contrast_range=0.15):
    return {
        "brightness": rng.uniform(-brightness_range, brightness_range),
        "contrast": 1.0 + rng.uniform(-contrast_range, contrast_range),
    }


def _apply_flip(img, params, is_mask=False):
    return np.fliplr(img) if params["flip"] else img


def _apply_rotation(img, params, is_mask=False):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), params["angle"], 1.0)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def _apply_brightness_contrast(img, params):
    """Only meaningful for the actual image, never for a binary mask."""
    img_range = img.max() - img.min() if img.max() > img.min() else 1.0
    shifted = img + params["brightness"] * img_range
    adjusted = (shifted - shifted.mean()) * params["contrast"] + shifted.mean()
    return adjusted


def augment_image(img: np.ndarray, seed: int = None) -> np.ndarray:
    """Apply random flip + rotation + brightness/contrast to a single image."""
    rng = np.random.default_rng(seed)
    flip_p = _random_flip_params(rng)
    rot_p = _random_rotation_params(rng)
    bc_p = _random_brightness_contrast_params(rng)

    out = _apply_flip(img, flip_p)
    out = _apply_rotation(out, rot_p)
    out = _apply_brightness_contrast(out, bc_p)
    return out


def augment_pair(img: np.ndarray, mask: np.ndarray, seed: int = None):
    """
    Apply the SAME random flip + rotation to both an image and its paired
    mask (for segmentation), but brightness/contrast ONLY to the image —
    a mask has no meaningful "brightness."
    """
    rng = np.random.default_rng(seed)
    flip_p = _random_flip_params(rng)
    rot_p = _random_rotation_params(rng)
    bc_p = _random_brightness_contrast_params(rng)

    img_out = _apply_flip(img, flip_p)
    img_out = _apply_rotation(img_out, rot_p, is_mask=False)
    img_out = _apply_brightness_contrast(img_out, bc_p)

    mask_out = _apply_flip(mask, flip_p)
    mask_out = _apply_rotation(mask_out, rot_p, is_mask=True)

    return img_out, mask_out