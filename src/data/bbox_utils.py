"""
Phase 7: convert a binary ROI mask into a bounding box.

Uses the bounding rectangle of ALL foreground (nonzero) pixels rather than
just the largest connected component — JPEG compression artifacts can
fragment a mask's edges into disconnected specks, and we'd rather keep the
full lesion extent than risk clipping it by only trusting one component.
"""

import cv2
import numpy as np


def mask_to_bbox(mask_img: np.ndarray, threshold: int = 127) -> dict:
    """
    Returns:
      'bbox': (x, y, w, h) in the mask's own pixel coordinates, or None if
              the mask is empty
      'fill_ratio': fraction of the bbox area actually covered by
              foreground pixels — a rough tightness/quality signal (a
              perfect rectangle-shaped lesion mask would be ~1.0; very low
              values suggest a scattered or degenerate mask)
    """
    mask_uint8 = _to_uint8(mask_img)
    _, binary = cv2.threshold(mask_uint8, threshold, 255, cv2.THRESH_BINARY)

    nonzero = cv2.findNonZero(binary)
    if nonzero is None:
        return {"bbox": None, "fill_ratio": 0.0}

    x, y, w, h = cv2.boundingRect(nonzero)
    n_fg_pixels = int(np.count_nonzero(binary))
    bbox_area = w * h
    fill_ratio = n_fg_pixels / bbox_area if bbox_area > 0 else 0.0

    return {"bbox": (int(x), int(y), int(w), int(h)), "fill_ratio": float(fill_ratio)}


def normalize_bbox(bbox: tuple, img_shape: tuple) -> tuple:
    """Convert (x, y, w, h) in pixels to normalized [0,1] (x, y, w, h),
    relative to img_shape = (height, width). Normalized coordinates survive
    resizing consistently -- always recompute pixel coords by multiplying
    by the CURRENT image's actual (height, width) at load time."""
    x, y, w, h = bbox
    img_h, img_w = img_shape[:2]
    return (x / img_w, y / img_h, w / img_w, h / img_h)


def denormalize_bbox(norm_bbox: tuple, img_shape: tuple) -> tuple:
    """Inverse of normalize_bbox."""
    nx, ny, nw, nh = norm_bbox
    img_h, img_w = img_shape[:2]
    return (nx * img_w, ny * img_h, nw * img_w, nh * img_h)


def transform_bbox_for_preprocessing(original_bbox: tuple, crop_bbox: tuple,
                                      resized_shape: tuple):
    """
    Carries a bounding box through the SAME background-removal crop + resize
    that preprocess_image() applies to the image itself (Phase 4). Without
    this, a box computed on the original full mammogram would point at the
    wrong location once the image has been cropped/resized.

    original_bbox: (x, y, w, h) in the ORIGINAL full-mammogram pixel coords
    crop_bbox:     (x0, y0, w0, h0) — the background-removal crop box,
                   exactly as returned by preprocess_image()['bbox']
    resized_shape: (height, width) — the final processed image size

    Returns (x, y, w, h) in the FINAL preprocessed image's pixel coords, or
    None if the box falls entirely outside the crop (rare — e.g. background
    removal failed to include the lesion region).
    """
    ox, oy, ow, oh = original_bbox
    cx, cy, cw, ch = crop_bbox

    # Shift into the crop's coordinate frame, then clip to the crop bounds.
    x0 = max(0, ox - cx)
    y0 = max(0, oy - cy)
    x1 = min(cw, ox - cx + ow)
    y1 = min(ch, oy - cy + oh)

    if x1 <= x0 or y1 <= y0:
        return None

    # Scale from crop size to the final resized size.
    scale_x = resized_shape[1] / cw if cw > 0 else 0
    scale_y = resized_shape[0] / ch if ch > 0 else 0

    rx0, ry0 = x0 * scale_x, y0 * scale_y
    rx1, ry1 = x1 * scale_x, y1 * scale_y

    return (rx0, ry0, rx1 - rx0, ry1 - ry0)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img_min, img_max = img.min(), img.max()
    denom = (img_max - img_min) if (img_max - img_min) > 1e-6 else 1.0
    return ((img - img_min) / denom * 255.0).astype(np.uint8)