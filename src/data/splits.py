"""
Phase 3, Dataset Rule #1: NEVER split by image/row — always split by patient.

If images from the same patient can land in both train and test, the model
can effectively "see" that patient during training (via anatomy, breast
density, imaging artifacts) and produce misleadingly high test performance.

This module:
  1. Aggregates each patient to a single label (did they have ANY malignant
     finding, across all their images?) so splitting can be stratified.
  2. Splits at the patient level using that aggregated label.
  3. Provides a hard assertion (`verify_no_patient_leakage`) that must pass
     before any split is used for training — call this in tests too
     (see tests/test_splits.py).
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def aggregate_patient_labels(df: pd.DataFrame,
                              patient_col: str = "patient_id",
                              label_col: str = "pathology_binary") -> pd.DataFrame:
    """One row per patient: malignant if ANY of their findings is malignant,
    else benign. This is the label used purely for stratifying the split —
    it is not necessarily the model's training target."""
    def patient_label(labels):
        return "malignant" if (labels == "malignant").any() else "benign"

    patient_labels = (
        df.groupby(patient_col)[label_col]
        .apply(patient_label)
        .reset_index()
        .rename(columns={label_col: "patient_level_label"})
    )
    return patient_labels


def patient_level_split(df: pd.DataFrame,
                         patient_col: str = "patient_id",
                         label_col: str = "pathology_binary",
                         test_size: float = 0.15,
                         val_size: float = 0.15,
                         seed: int = 42):
    """
    Split `df` into (train_df, val_df, test_df) such that every row for a
    given patient ends up in exactly one split.

    test_size / val_size are fractions of the *patient* population, not of
    the row count (a patient with many findings shouldn't count extra).
    """
    patient_labels = aggregate_patient_labels(df, patient_col, label_col)

    # Stage 1: carve out the test set of patients.
    train_val_patients, test_patients = train_test_split(
        patient_labels,
        test_size=test_size,
        stratify=patient_labels["patient_level_label"],
        random_state=seed,
    )

    # Stage 2: split the remainder into train/val. val_size is expressed as
    # a fraction of the ORIGINAL population, so rescale relative to what's left.
    relative_val_size = val_size / (1.0 - test_size)
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=relative_val_size,
        stratify=train_val_patients["patient_level_label"],
        random_state=seed,
    )

    train_ids = set(train_patients[patient_col])
    val_ids = set(val_patients[patient_col])
    test_ids = set(test_patients[patient_col])

    train_df = df[df[patient_col].isin(train_ids)].copy()
    val_df = df[df[patient_col].isin(val_ids)].copy()
    test_df = df[df[patient_col].isin(test_ids)].copy()

    verify_no_patient_leakage(train_df, val_df, test_df, patient_col)

    return train_df, val_df, test_df


def verify_no_patient_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               patient_col: str = "patient_id") -> None:
    """Raise if any patient appears in more than one split. This should
    never be skipped — it's the single most important check in Phase 3."""
    train_ids = set(train_df[patient_col])
    val_ids = set(val_df[patient_col])
    test_ids = set(test_df[patient_col])

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise AssertionError(
            "Patient leakage detected across splits!\n"
            f"  train∩val:  {overlap_train_val}\n"
            f"  train∩test: {overlap_train_test}\n"
            f"  val∩test:   {overlap_val_test}"
        )


def split_summary(train_df, val_df, test_df,
                   patient_col: str = "patient_id",
                   label_col: str = "pathology_binary") -> pd.DataFrame:
    """Small table of patient/row counts and class balance per split —
    goes straight into the EDA notebook / report."""
    rows = []
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rows.append({
            "split": name,
            "n_patients": split_df[patient_col].nunique(),
            "n_rows": len(split_df),
            "n_benign": (split_df[label_col] == "benign").sum(),
            "n_malignant": (split_df[label_col] == "malignant").sum(),
        })
    return pd.DataFrame(rows)