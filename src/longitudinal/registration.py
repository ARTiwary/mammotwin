"""
Phase 12: image registration — aligning a "prior" exam to the "current"
exam's coordinate frame before any pixel-level comparison is meaningful.

Uses ORB feature matching + RANSAC-robust partial-affine estimation
(rotation + translation + uniform scale) rather than a full projective
homography — a better match for what actually varies between mammography
visits (patient positioning), and more stable to estimate from a modest
number of feature matches. All via OpenCV, already a hard dependency —
no new third-party package, consistent with the project's dependency
caution after the Phase 7/8 issues.
"""

import cv2
import numpy as np


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img_min, img_max = img.min(), img.max()
    denom = (img_max - img_min) if (img_max - img_min) > 1e-6 else 1.0
    return ((img - img_min) / denom * 255.0).astype(np.uint8)


def register_images(moving_img: np.ndarray, fixed_img: np.ndarray, n_features: int = 2000):
    """
    Aligns moving_img onto fixed_img's coordinate frame.
    Returns (aligned_image, transform_matrix_or_None, n_inlier_matches).
    If registration fails (too few features/matches), returns the
    UNALIGNED moving image unchanged and None for the transform — callers
    should check for None and treat that case as "registration failed,"
    not silently trust an untransformed comparison.
    """
    moving_uint8 = _to_uint8(moving_img)
    fixed_uint8 = _to_uint8(fixed_img)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(moving_uint8, None)
    kp2, des2 = orb.detectAndCompute(fixed_uint8, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return moving_uint8.copy(), None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:200]

    if len(matches) < 4:
        return moving_uint8.copy(), None, 0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, inlier_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                                   ransacReprojThreshold=5.0)
    if M is None:
        return moving_uint8.copy(), None, 0

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    h, w = fixed_uint8.shape[:2]
    aligned = cv2.warpAffine(moving_uint8, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return aligned, M, n_inliers