"""
Phase 7 step 1: derive bounding-box ground truth from ROI masks.

For each row in metadata.csv, this:
  1. Finds all image candidates in the roi_mask_file_path's series UID
     folder (may be 1 or 2, per the known mask/cropped-patch UID-sharing
     quirk — see roi_utils.py).
  2. Classifies which candidate is the MASK (full-mammogram-sized) vs. the
     cropped patch, by comparing pixel dimensions.
  3. Extracts a bounding box from the mask.
  4. Saves everything to data/metadata/bbox_metadata.csv.

Usage (real data):
    python scripts/build_bbox_metadata.py --raw-images-dir "C:\\...\\Datasets\\jpeg" --limit 300

Demo mode:
    python scripts/build_bbox_metadata.py --demo
"""

import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.data.path_utils import get_all_candidates
from src.data.roi_utils import classify_roi_candidates
from src.data.bbox_utils import mask_to_bbox, normalize_bbox


def process_row(row, raw_images_dir):
    full_path = row.get("image_file_path_resolved")
    raw_mask_path = row.get("roi_mask_file_path")

    if pd.isna(full_path) or pd.isna(raw_mask_path) or not os.path.exists(full_path):
        return {"has_bbox": False, "reason": "missing_full_image_or_mask_path"}

    full_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if full_img is None:
        return {"has_bbox": False, "reason": "full_image_load_failed"}
    full_shape = full_img.shape

    candidates = get_all_candidates(raw_mask_path, raw_images_dir)
    if not candidates:
        return {"has_bbox": False, "reason": "no_mask_candidates_found"}

    classification = classify_roi_candidates(candidates, full_shape)
    mask_path = classification["mask_path"]
    if mask_path is None:
        return {"has_bbox": False, "reason": "could_not_identify_mask"}

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return {"has_bbox": False, "reason": "mask_load_failed"}

    if mask_img.shape != full_shape:
        return {"has_bbox": False, "reason": "mask_shape_mismatch",
                "mask_shape": mask_img.shape, "full_shape": full_shape}

    bbox_result = mask_to_bbox(mask_img)
    if bbox_result["bbox"] is None:
        return {"has_bbox": False, "reason": "empty_mask"}

    x, y, w, h = bbox_result["bbox"]
    nx, ny, nw, nh = normalize_bbox((x, y, w, h), full_shape)

    return {
        "has_bbox": True, "reason": "ok",
        "image_file_path_resolved": full_path,
        "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
        "bbox_x_norm": nx, "bbox_y_norm": ny, "bbox_w_norm": nw, "bbox_h_norm": nh,
        "fill_ratio": bbox_result["fill_ratio"],
        "mask_path_used": mask_path,
        "cropped_path_used": classification["cropped_path"],
        "ambiguous_roi_folder": classification["ambiguous"],
        "full_image_shape": f"{full_shape[0]}x{full_shape[1]}",
    }


def generate_demo_rows(n=15, seed=42):
    """Synthetic rows: each with a full image + matching full-size mask +
    small cropped patch sharing a folder, exactly like the real data."""
    rng = np.random.default_rng(seed)
    tmp_dir = "/tmp/mammotwin_phase7_demo"
    os.makedirs(tmp_dir, exist_ok=True)
    rows = []

    for i in range(n):
        full_h, full_w = int(rng.integers(800, 1200)), int(rng.integers(600, 900))
        full_img = rng.normal(50, 5, (full_h, full_w)).astype(np.uint8)
        full_path = os.path.join(tmp_dir, f"full_{i}.jpg")
        cv2.imwrite(full_path, full_img)

        bx, by = int(rng.integers(50, full_w - 150)), int(rng.integers(50, full_h - 150))
        bw, bh = int(rng.integers(40, 100)), int(rng.integers(40, 100))
        mask = np.zeros((full_h, full_w), dtype=np.uint8)
        mask[by:by + bh, bx:bx + bw] = 255

        cropped = rng.normal(150, 10, (bh + 20, bw + 20)).astype(np.uint8)

        folder = os.path.join(tmp_dir, f"uid_{i}")
        os.makedirs(folder, exist_ok=True)
        mask_path = os.path.join(folder, "1-100.jpg")
        cropped_path = os.path.join(folder, "2-101.jpg")
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(cropped_path, cropped)

        rows.append({
            "image_id": f"demo_{i}",
            "image_file_path_resolved": full_path,
            # raw_mask_path needs to resolve (via extract_series_uid) to folder "uid_i"
            "roi_mask_file_path": f"SomeCase_{i}/some_uid/uid_{i}/000000.dcm",
            "expected_bbox": (bx, by, bw, bh),
        })

    return pd.DataFrame(rows), tmp_dir


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 7: bbox ground truth")
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--raw-images-dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    config = load_config()
    metadata_dir = config["paths"]["data_metadata"]

    if args.demo:
        print("Running Phase 7 bbox extraction in DEMO mode.\n")
        df, raw_images_dir = generate_demo_rows()
    else:
        metadata_csv = args.metadata_csv or os.path.join(metadata_dir, "metadata.csv")
        if not os.path.exists(metadata_csv):
            print(f"No metadata CSV at {metadata_csv}. Run Phase 3 first, or pass --demo.")
            return
        if not args.raw_images_dir:
            print("Pass --raw-images-dir (your local jpeg folder).")
            return
        df = pd.read_csv(metadata_csv).head(args.limit)
        raw_images_dir = args.raw_images_dir
        print(f"Processing {len(df)} rows (--limit {args.limit})...\n")

    results = []
    for _, row in df.iterrows():
        result = process_row(row, raw_images_dir)
        result["image_id"] = row.get("image_id")
        result["patient_id"] = row.get("patient_id")
        results.append(result)

    result_df = pd.DataFrame(results)
    n_success = int(result_df["has_bbox"].sum())
    print(f"Bounding boxes extracted: {n_success}/{len(result_df)} "
          f"({100 * n_success / max(len(result_df), 1):.1f}%)\n")

    print("Failure reasons breakdown:")
    print(result_df[~result_df["has_bbox"]]["reason"].value_counts())

    if args.demo:
        # Verify correctness against known ground truth.
        merged = df.merge(result_df, on="image_id")
        n_correct = 0
        for _, row in merged.iterrows():
            if row["has_bbox"]:
                extracted = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])
                if extracted == row["expected_bbox"]:
                    n_correct += 1
        print(f"\nDemo self-check: {n_correct}/{n_success} extracted boxes exactly "
              f"match the known synthetic ground truth.")
        assert n_correct == n_success, "Some demo boxes did not match expected values!"
        print("Demo self-check PASSED.")

    out_path = os.path.join(metadata_dir, "demo_bbox_metadata.csv" if args.demo else "bbox_metadata.csv")
    os.makedirs(metadata_dir, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    n_ambiguous = int(result_df.get("ambiguous_roi_folder", pd.Series(dtype=bool)).sum())
    if n_ambiguous:
        print(f"NOTE: {n_ambiguous} rows had an ambiguous mask/crop folder — "
              f"worth spot-checking those manually.")


if __name__ == "__main__":
    main()