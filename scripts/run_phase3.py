"""
Phase 3 deliverable: build the unified metadata table, run integrity checks,
and produce locked patient-level train/val/test splits.

Real data mode (once you've downloaded CBIS-DDSM):
    python scripts/run_phase3.py \
        --mass-train path/to/mass_case_description_train_set.csv \
        --mass-test  path/to/mass_case_description_test_set.csv \
        --calc-train path/to/calc_case_description_train_set.csv \
        --calc-test  path/to/calc_case_description_test_set.csv \
        --raw-images-dir path/to/CBIS-DDSM/images

Demo mode (no real data needed — proves the split/leakage logic works):
    python scripts/run_phase3.py --demo
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config, set_global_seed
from src.data.build_metadata import build_unified_metadata
from src.data.splits import patient_level_split, split_summary, verify_no_patient_leakage
from src.data.data_checks import run_all_checks
from src.data.path_utils import add_resolved_paths


def generate_synthetic_metadata(n_patients: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Build a fake-but-realistically-shaped metadata table so the split and
    integrity-check logic can be demonstrated before real CBIS-DDSM data is
    downloaded. Clearly synthetic — never use this for actual training.
    """
    rng = np.random.default_rng(seed)
    views = ["CC", "MLO"]
    lateralities = ["LEFT", "RIGHT"]
    rows = []

    for i in range(n_patients):
        patient_id = f"P_{i:04d}"
        # Roughly 30% of patients have at least one malignant finding
        patient_is_malignant = rng.random() < 0.3
        n_findings = rng.integers(1, 4)  # some patients have multiple findings/images

        for f in range(n_findings):
            pathology = "MALIGNANT" if (patient_is_malignant and f == 0) else \
                        rng.choice(["BENIGN", "BENIGN_WITHOUT_CALLBACK"])
            rows.append({
                "patient_id": patient_id,
                "finding_type": rng.choice(["mass", "calc"]),
                "source_split": rng.choice(["train", "test"]),
                "laterality": rng.choice(lateralities),
                "view": rng.choice(views),
                "breast_density": rng.integers(1, 5),
                "abnormality_id": f,
                "abnormality_type": "mass",
                "pathology": pathology,
                "subtlety": rng.integers(1, 6),
                "image_file_path": f"synthetic/{patient_id}_{f}.dcm",
            })

    df = pd.DataFrame(rows)
    df["pathology"] = df["pathology"].str.upper()
    df["pathology_binary"] = df["pathology"].map({
        "BENIGN": "benign", "BENIGN_WITHOUT_CALLBACK": "benign", "MALIGNANT": "malignant",
    })
    df["image_id"] = (df["patient_id"] + "_" + df["finding_type"] + "_" +
                       df["abnormality_id"].astype(str) + "_" + df["view"])
    return df


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 3: metadata + splits")
    parser.add_argument("--mass-train", type=str, default=None)
    parser.add_argument("--mass-test", type=str, default=None)
    parser.add_argument("--calc-train", type=str, default=None)
    parser.add_argument("--calc-test", type=str, default=None)
    parser.add_argument("--raw-images-dir", type=str, default=None,
                         help="Your local 'jpeg' folder from the CBIS-DDSM download "
                              "(e.g. C:\\Users\\you\\Datasets\\CBIS-DDSM\\jpeg). "
                              "Used to resolve the CSV's DICOM-style paths to the "
                              "actual .jpg files on your machine.")
    parser.add_argument("--demo", action="store_true",
                         help="Run on synthetic data instead of real CBIS-DDSM CSVs")
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["project"]["seed"])
    metadata_dir = config["paths"]["data_metadata"]
    os.makedirs(metadata_dir, exist_ok=True)

    have_real_csvs = any([args.mass_train, args.mass_test, args.calc_train, args.calc_test])

    if args.demo or not have_real_csvs:
        print("No CBIS-DDSM CSVs provided (or --demo passed).")
        print("Running on SYNTHETIC metadata to demonstrate the pipeline.\n")
        metadata = generate_synthetic_metadata(seed=config["project"]["seed"])
        metadata_path = os.path.join(metadata_dir, "demo_synthetic_metadata.csv")
        check_files = False
    else:
        print("Building unified metadata from provided CBIS-DDSM CSVs...\n")
        metadata = build_unified_metadata(
            mass_train=args.mass_train, mass_test=args.mass_test,
            calc_train=args.calc_train, calc_test=args.calc_test,
        )
        metadata_path = os.path.join(metadata_dir, "metadata.csv")
        check_files = args.raw_images_dir is not None
        if args.raw_images_dir:
            print(f"Resolving CSV image paths against local root: {args.raw_images_dir}")
            metadata = add_resolved_paths(metadata, args.raw_images_dir)
            n_resolved = metadata["image_file_path_resolved"].notna().sum()
            n_ambiguous = (metadata["image_file_path_n_candidates"] > 1).sum()
            print(f"Resolved {n_resolved}/{len(metadata)} full-mammogram image paths "
                  f"({n_ambiguous} folders had more than one candidate image).")

    metadata.to_csv(metadata_path, index=False)
    print(f"Saved metadata table: {metadata_path} ({len(metadata)} rows, "
          f"{metadata['patient_id'].nunique()} patients)\n")

    print("=" * 60)
    print("DATA INTEGRITY CHECKS")
    print("=" * 60)
    # Once paths are resolved, check THOSE (image_file_path_resolved) — the
    # raw column still holds the original uploader's path and will always
    # look "missing" on your machine.
    path_col = "image_file_path_resolved" if (check_files and "image_file_path_resolved" in metadata.columns) else "image_file_path"
    run_all_checks(
        metadata,
        path_col=path_col,
        id_col="image_id",
        label_col="pathology_binary",
        base_dir=None,  # already resolved to absolute paths above
        check_files_exist=check_files,
    )

    print("\n" + "=" * 60)
    print("PATIENT-LEVEL SPLIT")
    print("=" * 60)
    train_df, val_df, test_df = patient_level_split(
        metadata, patient_col="patient_id", label_col="pathology_binary",
        test_size=0.15, val_size=0.15, seed=config["project"]["seed"],
    )

    # verify_no_patient_leakage already ran inside patient_level_split and
    # would have raised — this second call just makes the guarantee explicit
    # in the printed output.
    verify_no_patient_leakage(train_df, val_df, test_df, "patient_id")
    print("PASSED: no patient appears in more than one split.\n")

    summary = split_summary(train_df, val_df, test_df)
    print(summary.to_string(index=False))

    prefix = "demo_" if (args.demo or not have_real_csvs) else ""
    train_path = os.path.join(metadata_dir, f"{prefix}train_split.csv")
    val_path = os.path.join(metadata_dir, f"{prefix}val_split.csv")
    test_path = os.path.join(metadata_dir, f"{prefix}test_split.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved splits:\n  {train_path}\n  {val_path}\n  {test_path}")
    if args.demo or not have_real_csvs:
        print("\nNOTE: these are SYNTHETIC demo splits. Re-run with real CBIS-DDSM "
              "CSVs to produce data/metadata/{train,val,test}_split.csv for actual use.")
    else:
        print("\nTest set is now considered LOCKED. Do not tune on it until Phase 14.")


if __name__ == "__main__":
    main()