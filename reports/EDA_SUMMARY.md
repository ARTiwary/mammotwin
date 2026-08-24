# MammoTwin — Phase 5 EDA Summary

Dataset: 3568 rows, 1566 patients.

## Class distribution

| pathology_binary   |   count |
|:-------------------|--------:|
| benign             |    2111 |
| malignant          |    1457 |

Imbalance ratio (majority:minority): **1.45:1** — mild, standard class weighting should suffice.

## Image dimensions

Sampled 100 images (0 failed to load). Width range: (1846, 4306), height range: (3944, 6871).

## Shortcut / artifact audit

Cross-tabulated view, laterality, and breast density against pathology (see `data/metadata/phase5_crosstabs.csv`). Any field where one category shows a malignancy rate far outside the overall base rate is worth a second look before training — it may indicate an acquisition artifact the model could learn as a shortcut rather than genuine lesion signal.

Overall malignant rate: 40.8%

## Figures

- `phase5_class_distribution.png`
- `phase5_image_dimensions.png`
- `phase5_representative_cases.png`
- `phase5_cropped_examples.png`
