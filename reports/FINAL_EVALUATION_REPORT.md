# MammoTwin — Phase 14 Final Evaluation Report

Generated: 2026-09-02T22:55:28

Locked test set: 526 rows, 235 patients. This test set was untouched by any training or model-selection decision prior to this evaluation.

## Classification

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Balanced Acc. | Sensitivity | Specificity | Brier |
|---|---|---|---|---|---|---|
| whole_image_baseline | 0.752 [0.713, 0.796] | 0.697 [0.638, 0.755] | 0.654 | 0.530 | 0.778 | 0.209 |
| lesion_crop | 0.725 [0.680, 0.769] | 0.664 [0.596, 0.731] | 0.658 | 0.695 | 0.622 | 0.269 |
| multimodal | 0.758 [0.718, 0.799] | 0.715 [0.658, 0.774] | 0.675 | 0.521 | 0.830 | 0.208 |

## Localization

- map: 0.0639
- map_50: 0.1733
- map_75: 0.0301
- mean_top1_iou: 0.2981

## Segmentation

- dice: 0.1567
- iou: 0.0967

## Explainability (qualitative + overlap)

Mean fraction of ground-truth lesion box covered by high-attention heatmap pixels: 13.97%
See figure: C:\Users\21ayu\Desktop\memo\mammotwin\reports/figures\phase14_explainability_test_examples.png

## Robustness

Not evaluated — no external dataset was available for this project. Stated here as an explicit limitation.
