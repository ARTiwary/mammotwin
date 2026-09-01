"""
Filters a training metadata dataframe to exclude images that Phase 4's
quality gate flagged as unsuitable (blank, low-contrast, failed
background segmentation). This was flagged as a known gap since Phase 6
was first built and never actually wired in — fixed here.

Only applied to TRAINING data. Validation/test sets should NOT be
filtered this way — a model needs to be evaluated on the real,
unfiltered distribution it will actually see, quality issues included.
"""

import os
import pandas as pd


def filter_by_quality(df: pd.DataFrame, quality_report_path: str, id_col: str = "image_id") -> pd.DataFrame:
    """
    Returns df with rows removed where quality_report.csv marked
    passed=False for that image_id. If the quality report doesn't exist,
    or an image_id isn't in it (e.g. it wasn't part of the sample Phase 4
    was run on), that row is KEPT — this filter only removes rows we have
    positive evidence are bad, it never assumes badness from absence.
    """
    if not os.path.exists(quality_report_path):
        print(f"  NOTE: no quality report found at {quality_report_path} — "
              f"skipping quality-based filtering. Run scripts/run_phase4_preprocessing.py "
              f"(without --limit, or with a high one) to enable this.")
        return df

    quality_df = pd.read_csv(quality_report_path)
    if "passed" not in quality_df.columns or id_col not in quality_df.columns:
        print(f"  NOTE: {quality_report_path} doesn't have the expected columns — skipping filter.")
        return df

    failed_ids = set(quality_df.loc[quality_df["passed"] == False, id_col])  # noqa: E712
    if not failed_ids:
        print("  Quality filter: no flagged images found in the report — nothing to exclude.")
        return df

    n_before = len(df)
    filtered = df[~df[id_col].isin(failed_ids)].reset_index(drop=True)
    n_removed = n_before - len(filtered)
    print(f"  Quality filter: excluded {n_removed}/{n_before} training rows "
          f"flagged by Phase 4 (blank/low-contrast/failed segmentation).")
    return filtered