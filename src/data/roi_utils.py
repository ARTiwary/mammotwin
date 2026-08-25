"""
Phase 7: disambiguate ROI mask vs. cropped-lesion-patch images.

Per CBIS-DDSM's official documentation (Scientific Data paper, Lee et al.
2017): "Abnormalities are represented as binary mask images of the SAME
SIZE as their associated mammograms." Cropped-patch images, by contrast,
are small (mean ROI size ~450px per Scuccimarra 2018) — a tiny fraction of
a full mammogram's ~5000x3000 pixels.

CBIS-DDSM has a known quirk where the "cropped image" and "ROI mask" series
sometimes share the same series UID / folder, giving 2 files with no way to
tell which is which from the filename alone. We disambiguate by comparing
each candidate's pixel area to the FULL mammogram's area: whichever is
close to full-image size is the mask; the much smaller one is the crop.
"""

import cv2
import numpy as np


def classify_roi_candidates(candidate_paths: list, full_image_shape: tuple,
                             size_match_tolerance: float = 0.5) -> dict:
    """
    candidate_paths: list of image file paths found in one series UID folder
                      (from path_utils.get_all_candidates)
    full_image_shape: (height, width) of the FULL mammogram for this row
                       (i.e. the shape of image_file_path_resolved)
    size_match_tolerance: a candidate is considered "full-image-sized" (and
                      therefore the MASK) if its area is within this
                      fraction of the full image's area (default 50%, since
                      masks are typically byte-identical resolution to the
                      mammogram, but we allow slack for edge-case exports).

    Returns {'mask_path': str_or_None, 'cropped_path': str_or_None,
             'ambiguous': bool}
    """
    if not candidate_paths:
        return {"mask_path": None, "cropped_path": None, "ambiguous": False}

    full_area = full_image_shape[0] * full_image_shape[1]

    sized_candidates = []
    for path in candidate_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        area = img.shape[0] * img.shape[1]
        sized_candidates.append((path, area))

    if not sized_candidates:
        return {"mask_path": None, "cropped_path": None, "ambiguous": False}

    if len(sized_candidates) == 1:
        path, area = sized_candidates[0]
        is_full_sized = area >= full_area * (1 - size_match_tolerance)
        if is_full_sized:
            return {"mask_path": path, "cropped_path": None, "ambiguous": False}
        return {"mask_path": None, "cropped_path": path, "ambiguous": False}

    # Multiple candidates: the one with area closest to full_area is the
    # mask, the smallest is the cropped patch. Flag ambiguous if BOTH are
    # far from expected sizes (both small, or both full-sized) — unusual
    # and worth a human glance.
    sized_candidates.sort(key=lambda t: t[1], reverse=True)  # largest first
    largest_path, largest_area = sized_candidates[0]
    smallest_path, smallest_area = sized_candidates[-1]

    largest_is_full_sized = largest_area >= full_area * (1 - size_match_tolerance)
    smallest_is_small = smallest_area < full_area * (1 - size_match_tolerance)
    ambiguous = not (largest_is_full_sized and smallest_is_small)

    return {"mask_path": largest_path, "cropped_path": smallest_path, "ambiguous": ambiguous}