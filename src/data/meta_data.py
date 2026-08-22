"""
Phase 3: build a unified metadata table from CBIS-DDSM's official CSVs.

CBIS-DDSM ships four case-description CSVs (from TCIA or the Kaggle mirror):
  - mass_case_description_train_set.csv
  - mass_case_description_test_set.csv
  - calc_case_description_train_set.csv
  - calc_case_description_test_set.csv

IMPORTANT: we do NOT trust CBIS-DDSM's own train/test split blindly. Some
distributions have the same patient appearing in both "train" and "test"
CSVs. We load the official split only as `source_split` metadata for
reference; the actual split used for modeling is produced independently in
src/data/splits.py, with a patient-overlap check to guarantee no leakage.

Usage:
    python -m src.data.build_metadata \
        --mass-train path/to/mass_case_description_train_set.csv \
        --mass-test  path/to/mass_case_description_test_set.csv \
        --calc-train path/to/calc_case_description_train_set.csv \
        --calc-test  path/to/calc_case_description_test_set.csv \
        --output data/metadata/metadata.csv
"""

import argparse
import os
import pandas as pd

# Column names differ slightly between mass/calc CSVs in the official
# distribution. This mapping normalizes both onto one common schema.
COLUMN_MAP = {
    "patient_id": "patient_id",
    "breast_density": "breast_density",
    "breast density": "breast_density",       # calc CSVs sometimes use a space
    "left or right breast": "laterality",
    "image view": "view",
    "abnormality id": "abnormality_id",
    "abnormality type": "abnormality_type",
    "mass shape": "mass_shape",
    "mass margins": "mass_margins",
    "calc type": "calc_type",
    "calc distribution": "calc_distribution",
    "assessment": "assessment",
    "pathology": "pathology",
    "subtlety": "subtlety",
    "image file path": "image_file_path",
    "cropped image file path": "cropped_image_file_path",
    "ROI mask file path": "roi_mask_file_path",
}

# Columns every unified row should have, even if the source CSV lacks some
# (e.g. calc-specific columns will be NaN for mass rows and vice versa).
UNIFIED_COLUMNS = [
    "patient_id", "finding_type", "source_split", "laterality", "view",
    "breast_density", "abnormality_id", "abnormality_type",
    "mass_shape", "mass_margins", "calc_type", "calc_distribution",
    "assessment", "pathology", "subtlety",
    "image_file_path", "cropped_image_file_path", "roi_mask_file_path",
]

# Raw CBIS-DDSM pathology labels collapsed into a clean binary target.
# BENIGN_WITHOUT_CALLBACK is a benign finding that did not require
# follow-up -> grouped with benign for the binary target, kept distinct
# in `pathology` (the original, ungrouped label) for anyone who wants the
# finer-grained distinction later.
PATHOLOGY_TO_BINARY = {
    "BENIGN": "benign",
    "BENIGN_WITHOUT_CALLBACK": "benign",
    "MALIGNANT": "malignant",
}


def load_cbis_csv(path: str, finding_type: str, source_split: str) -> pd.DataFrame:
    """Load one CBIS-DDSM CSV and normalize it onto the unified schema."""
    df = pd.read_csv(path)
    df = df.rename(columns={c: COLUMN_MAP.get(c, c) for c in df.columns})

    df["finding_type"] = finding_type       # "mass" or "calc"
    df["source_split"] = source_split       # "train" or "test", per CBIS-DDSM's own split

    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[UNIFIED_COLUMNS]


def build_unified_metadata(mass_train=None, mass_test=None,
                            calc_train=None, calc_test=None) -> pd.DataFrame:
    """Load whichever of the four CBIS-DDSM CSVs are provided and concatenate."""
    frames = []
    sources = [
        (mass_train, "mass", "train"),
        (mass_test, "mass", "test"),
        (calc_train, "calc", "train"),
        (calc_test, "calc", "test"),
    ]
    for path, finding_type, split in sources:
        if path is not None:
            frames.append(load_cbis_csv(path, finding_type, split))

    if not frames:
        raise ValueError("No CSV paths provided — pass at least one of "
                          "--mass-train/--mass-test/--calc-train/--calc-test")

    metadata = pd.concat(frames, ignore_index=True)

    # Clean/standardize pathology labels and derive the binary target.
    metadata["pathology"] = metadata["pathology"].astype(str).str.strip().str.upper()
    metadata["pathology_binary"] = metadata["pathology"].map(PATHOLOGY_TO_BINARY)

    # A stable per-row image ID, useful for joining against files on disk.
    metadata["image_id"] = (
        metadata["patient_id"].astype(str) + "_" +
        metadata["finding_type"] + "_" +
        metadata["abnormality_id"].astype(str) + "_" +
        metadata["view"].astype(str)
    )

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Build unified CBIS-DDSM metadata table")
    parser.add_argument("--mass-train", type=str, default=None)
    parser.add_argument("--mass-test", type=str, default=None)
    parser.add_argument("--calc-train", type=str, default=None)
    parser.add_argument("--calc-test", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/metadata/metadata.csv")
    args = parser.parse_args()

    metadata = build_unified_metadata(
        mass_train=args.mass_train, mass_test=args.mass_test,
        calc_train=args.calc_train, calc_test=args.calc_test,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    metadata.to_csv(args.output, index=False)

    print(f"Unified metadata saved to: {args.output}")
    print(f"Total rows: {len(metadata)}")
    print(f"Unique patients: {metadata['patient_id'].nunique()}")
    print("\nPathology (binary) distribution:")
    print(metadata["pathology_binary"].value_counts(dropna=False))
    print("\nFinding type distribution:")
    print(metadata["finding_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()