"""
Phase 3 data integrity checks: missing/corrupt files, duplicates, class
balance. Run this on the unified metadata table before splitting, and again
on each split before training.
"""

import os
import pandas as pd


def check_missing_files(df: pd.DataFrame, path_col: str,
                         base_dir: str = None) -> pd.DataFrame:
    """Return the subset of rows whose referenced file does not exist on disk.
    Pass base_dir if the CSV's paths are relative to the raw data folder."""
    def resolve(p):
        if pd.isna(p):
            return None
        return os.path.join(base_dir, p) if base_dir else p

    missing_mask = df[path_col].apply(
        lambda p: (resolve(p) is None) or (not os.path.exists(resolve(p)))
    )
    return df[missing_mask]


def check_corrupt_images(paths, base_dir: str = None) -> list:
    """Attempt to open each image; return the list of paths that fail.
    Requires OpenCV. Run this only once files actually exist on disk."""
    import cv2
    corrupt = []
    for p in paths:
        full_path = os.path.join(base_dir, p) if base_dir else p
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            corrupt.append(p)
    return corrupt


def check_duplicate_ids(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Return rows whose id_col value appears more than once."""
    dup_mask = df.duplicated(subset=[id_col], keep=False)
    return df[dup_mask].sort_values(id_col)


def class_distribution(df: pd.DataFrame, label_col: str) -> pd.Series:
    return df[label_col].value_counts(dropna=False)


def run_all_checks(df: pd.DataFrame, path_col: str = "image_file_path",
                    id_col: str = "image_id", label_col: str = "pathology_binary",
                    base_dir: str = None, check_files_exist: bool = True) -> dict:
    """Run the full Phase 3 checklist and print a summary report."""
    report = {}

    print("=== Class distribution ===")
    dist = class_distribution(df, label_col)
    print(dist)
    report["class_distribution"] = dist

    print("\n=== Duplicate IDs ===")
    dupes = check_duplicate_ids(df, id_col)
    print(f"Found {len(dupes)} duplicate rows" if len(dupes) else "No duplicates found")
    report["duplicates"] = dupes

    if check_files_exist:
        print("\n=== Missing files ===")
        missing = check_missing_files(df, path_col, base_dir)
        print(f"Found {len(missing)} rows with missing files" if len(missing) else "All files found")
        report["missing_files"] = missing
    else:
        print("\n=== Missing files === (skipped — no base_dir / raw files available yet)")

    print("\n=== Patients per view/laterality ===")
    if "view" in df.columns and "laterality" in df.columns:
        print(df.groupby(["laterality", "view"]).size())

    return report