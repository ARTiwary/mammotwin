"""
Phase 13: preprocessing for the structured/tabular branch of the
multimodal model.

Handles missing values carefully, per the plan's explicit requirement:
  - Categorical columns: an explicit "missing" category is used rather than
    imputing a mode — for CBIS-DDSM specifically, mass_shape/mass_margins
    are missing for calcification findings and calc_type/calc_distribution
    are missing for mass findings BY DEFINITION (a finding is one or the
    other), so "missing" here is informative (it tells you the finding
    type), not a data-quality gap to be papered over.
  - Numeric columns: median-imputed using ONLY the training set's median
    (never val/test — that would leak test-set information into training),
    plus an explicit "was_missing" indicator column, then standardized
    using the training set's mean/std.

All fitting happens on the TRAIN split only; val/test are transformed
using those fitted statistics — the same train/inference-consistency
principle used throughout this project since Phase 4.
"""

import numpy as np
import pandas as pd


class TabularPreprocessor:
    def __init__(self, categorical_cols: list, numeric_cols: list):
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.vocabularies = {}     # col -> list of known categories (+ "missing")
        self.numeric_medians = {}  # col -> median from TRAIN
        self.numeric_means = {}
        self.numeric_stds = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        for col in self.categorical_cols:
            values = df[col].fillna("missing").astype(str)
            vocab = sorted(values.unique().tolist())
            if "missing" not in vocab:
                vocab.append("missing")
            self.vocabularies[col] = vocab

        for col in self.numeric_cols:
            median = df[col].median()
            self.numeric_medians[col] = median
            filled = df[col].fillna(median)
            self.numeric_means[col] = filled.mean()
            self.numeric_stds[col] = filled.std() if filled.std() > 1e-6 else 1.0

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() on the TRAINING set before transform().")

        feature_blocks = []

        for col in self.categorical_cols:
            values = df[col].fillna("missing").astype(str)
            vocab = self.vocabularies[col]
            # Anything not seen during fit (a truly novel category at
            # inference time) safely falls back to "missing" rather than
            # crashing or silently misassigning it to an arbitrary column.
            values = values.apply(lambda v: v if v in vocab else "missing")
            one_hot = pd.get_dummies(values).reindex(columns=vocab, fill_value=0)
            feature_blocks.append(one_hot.values.astype(np.float32))

        for col in self.numeric_cols:
            was_missing = df[col].isna().astype(np.float32).values.reshape(-1, 1)
            filled = df[col].fillna(self.numeric_medians[col])
            standardized = (filled - self.numeric_means[col]) / self.numeric_stds[col]
            feature_blocks.append(standardized.values.astype(np.float32).reshape(-1, 1))
            feature_blocks.append(was_missing)

        return np.concatenate(feature_blocks, axis=1)

    @property
    def output_dim(self) -> int:
        dim = sum(len(v) for v in self.vocabularies.values())
        dim += 2 * len(self.numeric_cols)  # value + was_missing per numeric col
        return dim