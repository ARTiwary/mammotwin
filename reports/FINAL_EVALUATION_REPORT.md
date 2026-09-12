# MammoTwin — Phase 14 Final Evaluation Report

Generated: 2026-09-12T23:51:28

Locked test set: 526 rows, 235 patients. This test set was untouched by any training or model-selection decision prior to this evaluation.

## Classification

ROC-AUC/PR-AUC are threshold-independent (rank-based) and unaffected by the cutoff below. Sensitivity/specificity/balanced accuracy ARE threshold-dependent and are reported at TWO cutoffs so the effect of proper threshold selection is visible rather than hidden: the untuned default (0.5) that earlier reports used, and the operating threshold selected on the VALIDATION set only (see `select_operating_thresholds.py`).

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier | Sens. @0.5 | Spec. @0.5 | Operating thr. | Sens. @thr. | Spec. @thr. | Bal.Acc. @thr. | Thr. selected on |
|---|---|---|---|---|---|---|---|---|---|---|
| whole_image_baseline | 0.752 [0.713, 0.796] | 0.697 [0.638, 0.755] | 0.209 | 0.530 | 0.778 | 0.160 | 0.874 | 0.450 | 0.662 | val (target 90%) |
| lesion_crop | 0.725 [0.680, 0.769] | 0.664 [0.596, 0.731] | 0.269 | 0.695 | 0.622 | 0.048 | 0.915 | 0.281 | 0.598 | val (target 90%) |
| multimodal | 0.758 [0.718, 0.799] | 0.715 [0.658, 0.774] | 0.208 | 0.521 | 0.830 | 0.053 | 0.916 | 0.318 | 0.617 | val (target 90%) |

## Localization

- map: 0.0575
- map_50: 0.1530
- map_75: 0.0268
- mean_top1_iou: 0.2069

## Segmentation

- dice: 0.1605
- iou: 0.1042

## Explainability (qualitative + overlap)

Mean fraction of ground-truth lesion box covered by high-attention malignant-class Grad-CAM pixels: 4.98%  95% CI [3.09%, 7.17%]  (n=200)

Grad-CAM is generated for the malignant-class score specifically, regardless of the model's own top prediction on a given image, so this number consistently answers "where does the model see evidence of malignancy" rather than mixing that question with "evidence for whatever this model happened to predict."
See figure: C:\Users\21ayu\Desktop\memo\mammotwin\reports/figures\phase14_explainability_test_examples.png

## Robustness

Not evaluated — no external dataset was available for this project. Stated here as an explicit limitation.
