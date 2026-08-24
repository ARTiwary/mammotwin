"""
Phase 4 deliverable: run the full preprocessing pipeline (background
removal -> quality checks -> normalize -> resize) across the dataset,
producing:
  1. data/metadata/quality_report.csv — one row per image with quality flags
  2. reports/figures/phase4_preprocessing_examples.png — before/after grid

Real data mode:
    python scripts/run_phase4_preprocessing.py --limit 200
        (reads data/metadata/train_split.csv by default; --limit caps how
         many rows to process, since running all ~3500 can take a while)

Demo mode (no real data needed — exercises every quality-check branch):
    python scripts/run_phase4_preprocessing.py --demo
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import load_config, set_global_seed
from src.data.image_io import load_image, generate_synthetic_mammogram
from src.preprocessing.basic_preprocess import preprocess_image, save_preprocessing_config


def generate_demo_images(seed: int = 42) -> dict:
    """A handful of synthetic test cases that exercise every quality-check
    branch: a normal image, a blank image, a low-contrast image, and a
    normal image with a bright label artifact in the corner."""
    rng = np.random.default_rng(seed)

    normal = generate_synthetic_mammogram(size=512, seed=seed)

    blank = np.full((512, 512), fill_value=20, dtype=np.float32)
    blank += rng.normal(0, 0.5, size=blank.shape).astype(np.float32)  # near-zero noise

    low_contrast = generate_synthetic_mammogram(size=512, seed=seed + 1)
    low_contrast = 100 + (low_contrast - low_contrast.mean()) * 0.15  # compress dynamic range, but keep enough variance to not also trigger is_blank

    with_artifact = generate_synthetic_mammogram(size=512, seed=seed + 2).copy()
    with_artifact[10:40, 460:500] = 255  # bright label patch, small vs. breast

    return {
        "normal": normal,
        "blank": blank,
        "low_contrast": low_contrast,
        "with_label_artifact": with_artifact,
    }


def process_one(image_id, raw_img, config):
    result = preprocess_image(raw_img, config, run_quality_gate=True)
    row = {
        "image_id": image_id,
        "breast_area_fraction": result["breast_area_fraction"],
        **result["quality"],
    }
    return row, result


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 4: preprocessing + quality checks")
    parser.add_argument("--metadata-csv", type=str, default=None,
                         help="Metadata CSV to read (default: data/metadata/train_split.csv)")
    parser.add_argument("--path-col", type=str, default="image_file_path_resolved",
                         help="Column holding the resolved local image path")
    parser.add_argument("--limit", type=int, default=200,
                         help="Max number of images to process (real-data mode only)")
    parser.add_argument("--demo", action="store_true",
                         help="Run on synthetic images instead of real data")
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["project"]["seed"])
    metadata_dir = config["paths"]["data_metadata"]
    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    quality_rows = []
    examples = []  # list of (label, raw_img, processed_img) for the figure

    if args.demo:
        print("Running Phase 4 in DEMO mode on synthetic test images.\n")
        demo_images = generate_demo_images(seed=config["project"]["seed"])
        for image_id, raw_img in demo_images.items():
            row, result = process_one(image_id, raw_img, config)
            quality_rows.append(row)
            examples.append((image_id, raw_img, result["processed"]))
            status = "PASSED" if row["passed"] else "FLAGGED"
            print(f"  {image_id:22s} -> {status}  (flags: "
                  f"{ {k: v for k, v in row.items() if k not in ('image_id', 'passed', 'breast_area_fraction')} })")

    else:
        metadata_csv = args.metadata_csv or os.path.join(metadata_dir, "train_split.csv")
        if not os.path.exists(metadata_csv):
            print(f"No metadata CSV found at {metadata_csv}.")
            print("Run Phase 3 first, or pass --demo to test on synthetic images.")
            return

        df = pd.read_csv(metadata_csv)
        if args.path_col not in df.columns:
            print(f"Column '{args.path_col}' not found in {metadata_csv}.")
            print(f"Available columns: {list(df.columns)}")
            return

        df = df.dropna(subset=[args.path_col]).head(args.limit)
        print(f"Processing {len(df)} images from {metadata_csv} "
              f"(limited to --limit {args.limit})...\n")

        n_shown = 0
        for _, row_data in df.iterrows():
            image_id = row_data.get("image_id", row_data[args.path_col])
            path = row_data[args.path_col]
            try:
                raw_img = load_image(path)
            except Exception as e:
                quality_rows.append({"image_id": image_id, "load_error": str(e),
                                      "passed": False})
                continue

            row, result = process_one(image_id, raw_img, config)
            quality_rows.append(row)

            # Grab a few examples for the figure: first couple passing,
            # and any flagged ones we encounter (more informative than
            # showing 4 identical-looking passing cases).
            if (not row["passed"]) or n_shown < 2:
                examples.append((str(image_id)[:20], raw_img, result["processed"]))
                n_shown += 1

    # --- Save quality report ---
    quality_df = pd.DataFrame(quality_rows)
    report_path = os.path.join(metadata_dir, "quality_report.csv")
    quality_df.to_csv(report_path, index=False)

    n_total = len(quality_df)
    n_passed = int(quality_df["passed"].sum()) if "passed" in quality_df.columns else 0
    print(f"\nQuality report saved: {report_path}")
    print(f"Passed: {n_passed}/{n_total}  ({100 * n_passed / max(n_total, 1):.1f}%)")
    if "passed" in quality_df.columns and n_total > n_passed:
        print("\nFlag breakdown among failing images:")
        for col in quality_df.columns:
            if col in ("image_id", "passed", "breast_area_fraction"):
                continue
            if quality_df[col].dtype == bool:
                print(f"  {col}: {int(quality_df[col].sum())}")

    # --- Save before/after figure ---
    if examples:
        n = min(len(examples), 4)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)
        for i, (label, raw_img, processed_img) in enumerate(examples[:n]):
            axes[0, i].imshow(raw_img, cmap="gray")
            axes[0, i].set_title(f"Raw: {label}")
            axes[0, i].axis("off")
            axes[1, i].imshow(processed_img, cmap="gray")
            axes[1, i].set_title("Preprocessed")
            axes[1, i].axis("off")
        plt.tight_layout()
        fig_path = os.path.join(figures_dir, "phase4_preprocessing_examples.png")
        plt.savefig(fig_path, dpi=150)
        print(f"\nSaved before/after figure: {fig_path}")

    # --- Save the exact preprocessing config used, for provenance ---
    config_record_path = os.path.join(metadata_dir, "preprocessing_config_used.json")
    save_preprocessing_config(config, config_record_path)
    print(f"Saved preprocessing config record: {config_record_path}")


if __name__ == "__main__":
    main()