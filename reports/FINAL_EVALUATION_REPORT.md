# MammoTwin — Phase 14 Final Evaluation Report

Generated: 2026-08-30T00:31:56

Locked test set: 526 rows, 235 patients. This test set was untouched by any training or model-selection decision prior to this evaluation.

## Classification

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Balanced Acc. | Sensitivity | Specificity | Brier |
|---|---|---|---|---|---|---|
| whole_image_baseline | 0.678 [0.627, 0.722] | 0.573 [0.506, 0.648] | 0.637 | 0.744 | 0.531 | 0.305 |
| lesion_crop | 0.696 [0.648, 0.743] | 0.615 [0.545, 0.693] | 0.643 | 0.577 | 0.709 | 0.281 |
| multimodal | 0.677 [0.632, 0.724] | 0.586 [0.523, 0.657] | 0.614 | 0.823 | 0.405 | 0.261 |

## Localization

- map: 0.0639
- map_50: 0.1734
- map_75: 0.0300
- mean_top1_iou: 0.2981

## Segmentation

- dice: 0.1793
- iou: 0.1153

## Explainability (qualitative + overlap)

Mean fraction of ground-truth lesion box covered by high-attention heatmap pixels: 0.00%
See figure: C:\Users\21ayu\Desktop\memo\mammotwin\reports/figures\phase14_explainability_test_examples.png

## Robustness

Not evaluated — no external dataset was available for this project. Stated here as an explicit limitation.
