"""
Phase 16: tests for the patient-level split logic (src/data/splits.py) --
this is "the single most important check in Phase 3" per that module's own
docstring, and Phase 16 explicitly calls out "Test patient-level split
logic" as required.
"""
import pandas as pd
import numpy as np
import pytest

from src.data.splits import (
    aggregate_patient_labels,
    patient_level_split,
    verify_no_patient_leakage,
    split_summary,
)


def make_fake_dataset(n_patients=40, seed=0):
    """Each patient has 1-4 rows (images), matching real CBIS-DDSM's
    structure where one patient contributes multiple views/lesions."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        n_images = rng.integers(1, 5)
        # Roughly a third of patients have at least one malignant finding.
        patient_is_malignant = rng.random() < 0.35
        for img_i in range(n_images):
            if patient_is_malignant and img_i == 0:
                label = "malignant"
            else:
                label = rng.choice(["benign", "malignant"], p=[0.85, 0.15])
            rows.append({"patient_id": f"P{pid:03d}", "image_id": f"P{pid:03d}_{img_i}",
                         "pathology_binary": label})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# aggregate_patient_labels
# ---------------------------------------------------------------------

def test_aggregate_patient_labels_one_row_per_patient():
    df = make_fake_dataset()
    labels = aggregate_patient_labels(df)
    assert len(labels) == df["patient_id"].nunique()


def test_aggregate_patient_labels_malignant_if_any_finding_is():
    df = pd.DataFrame({
        "patient_id": ["A", "A", "B", "B"],
        "pathology_binary": ["benign", "malignant", "benign", "benign"],
    })
    labels = aggregate_patient_labels(df).set_index("patient_id")["patient_level_label"]
    assert labels["A"] == "malignant"
    assert labels["B"] == "benign"


# ---------------------------------------------------------------------
# patient_level_split -- the critical leakage-prevention property
# ---------------------------------------------------------------------

def test_patient_level_split_has_no_patient_overlap():
    df = make_fake_dataset(n_patients=60, seed=1)
    train_df, val_df, test_df = patient_level_split(df, test_size=0.15, val_size=0.15, seed=42)

    train_ids = set(train_df["patient_id"])
    val_ids = set(val_df["patient_id"])
    test_ids = set(test_df["patient_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_patient_level_split_covers_every_row_exactly_once():
    df = make_fake_dataset(n_patients=50, seed=2)
    train_df, val_df, test_df = patient_level_split(df, test_size=0.15, val_size=0.15, seed=42)

    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)

    all_image_ids = set(train_df["image_id"]) | set(val_df["image_id"]) | set(test_df["image_id"])
    assert all_image_ids == set(df["image_id"])


def test_patient_level_split_is_reproducible_with_same_seed():
    df = make_fake_dataset(n_patients=50, seed=3)
    t1, v1, s1 = patient_level_split(df, seed=7)
    t2, v2, s2 = patient_level_split(df, seed=7)
    assert set(t1["patient_id"]) == set(t2["patient_id"])
    assert set(v1["patient_id"]) == set(v2["patient_id"])
    assert set(s1["patient_id"]) == set(s2["patient_id"])


def test_patient_level_split_approximate_proportions():
    """Split sizes are expressed as fractions of PATIENTS, not rows -- with
    enough patients, the realized proportions should be close to requested."""
    df = make_fake_dataset(n_patients=200, seed=4)
    train_df, val_df, test_df = patient_level_split(df, test_size=0.15, val_size=0.15, seed=42)

    n_total_patients = df["patient_id"].nunique()
    test_frac = test_df["patient_id"].nunique() / n_total_patients
    val_frac = val_df["patient_id"].nunique() / n_total_patients

    assert test_frac == pytest.approx(0.15, abs=0.03)
    assert val_frac == pytest.approx(0.15, abs=0.03)


# ---------------------------------------------------------------------
# verify_no_patient_leakage -- must actually catch a deliberately broken split
# ---------------------------------------------------------------------

def test_verify_no_patient_leakage_passes_on_clean_split():
    df = make_fake_dataset(n_patients=30, seed=5)
    train_df, val_df, test_df = patient_level_split(df, seed=42)
    verify_no_patient_leakage(train_df, val_df, test_df)  # should not raise


def test_verify_no_patient_leakage_catches_deliberately_introduced_leakage():
    df = make_fake_dataset(n_patients=30, seed=6)
    train_df, val_df, test_df = patient_level_split(df, seed=42)

    # Deliberately introduce a leak: copy one training patient's rows into test.
    leaking_patient = train_df["patient_id"].iloc[0]
    leaked_rows = train_df[train_df["patient_id"] == leaking_patient]
    contaminated_test_df = pd.concat([test_df, leaked_rows], ignore_index=True)

    with pytest.raises(AssertionError, match="leakage"):
        verify_no_patient_leakage(train_df, val_df, contaminated_test_df)


# ---------------------------------------------------------------------
# split_summary
# ---------------------------------------------------------------------

def test_split_summary_row_counts_match():
    df = make_fake_dataset(n_patients=40, seed=8)
    train_df, val_df, test_df = patient_level_split(df, seed=42)
    summary = split_summary(train_df, val_df, test_df)

    assert set(summary["split"]) == {"train", "val", "test"}
    for _, row in summary.iterrows():
        assert row["n_benign"] + row["n_malignant"] == row["n_rows"]
