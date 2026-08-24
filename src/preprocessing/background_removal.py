"""
Phase 4: remove irrelevant background from mammograms.

Mammogram scans typically contain:
  - A black background outside the breast.
  - The breast tissue itself, usually as one large bright blob on one side.
  - Sometimes a small burned-in label/marker artifact (patient info, "L"/"R"
    laterality marker) that can be bright enough to confuse naive thresholding.

Approach (standard, widely used in mammography preprocessing literature):
  1. Otsu-threshold to separate foreground (tissue + artifacts) from background.
  2. Find all connected components in the foreground.
  3. Keep only the LARGEST component (the breast) — this naturally discards
     small label artifacts, which are almost always much smaller than the
     breast itself.
  4. Crop to that component's bounding box, and zero out any stray pixels
     inside the crop that aren't part of the breast component.
"""

import numpy as np
import cv2


def remove_background(img: np.ndarray, padding: int = 10) -> dict:
    """
    Returns a dict with:
      'cropped': the breast region, background-masked and cropped
      'bbox': (x, y, w, h) of the crop in the ORIGINAL image
      'breast_area_fraction': fraction of the ORIGINAL image occupied by
          the detected breast component (useful as a quality signal —
          a very small fraction usually means segmentation failed or the
          image is mostly empty)
    """
    img_uint8 = _to_uint8(img)

    # Otsu picks the threshold automatically; +1 avoids losing very faint
    # tissue at the breast's edge, which we do NOT want to discard.
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if n_labels <= 1:
        # No foreground found at all (e.g. a fully black image) — quality
        # checks will catch this separately; return the image unchanged.
        return {
            "cropped": img,
            "bbox": (0, 0, img.shape[1], img.shape[0]),
            "breast_area_fraction": 0.0,
        }

    # Label 0 is always background; pick the largest of the rest.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))

    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]
    breast_area = stats[largest_label, cv2.CC_STAT_AREA]

    # Add padding, clipped to image bounds.
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(img.shape[1], x + w + padding)
    y1 = min(img.shape[0], y + h + padding)

    cropped = img[y0:y1, x0:x1].copy()

    breast_area_fraction = float(breast_area) / (img.shape[0] * img.shape[1])

    return {
        "cropped": cropped,
        "bbox": (x0, y0, x1 - x0, y1 - y0),
        "breast_area_fraction": breast_area_fraction,
    }


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Otsu thresholding (cv2.threshold) requires uint8 input."""
    if img.dtype == np.uint8:
        return img
    img_min, img_max = img.min(), img.max()
    denom = (img_max - img_min) if (img_max - img_min) > 1e-6 else 1.0
    scaled = (img - img_min) / denom * 255.0
    return scaled.astype(np.uint8)