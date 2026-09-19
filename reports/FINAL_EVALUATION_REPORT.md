# MammoTwin — Phase 14 Final Evaluation Report

Generated: 2026-09-19T23:06:37

Locked test set: 526 rows, 235 patients. This test set was untouched by any training or model-selection decision prior to this evaluation.

## Dataset Split

Split at the PATIENT level (see `src/data/splits.py`) -- no patient appears in more than one split, preventing the same patient's images from leaking across train/val/test.

| Split | Patients | % of patients | Images (rows) | % of rows | Benign | Malignant |
|---|---|---|---|---|---|---|
| train | 1096 | 70.0% | 2501 | 70.1% | 1470 | 1031 |
| val | 235 | 15.0% | 541 | 15.2% | 330 | 211 |
| test | 235 | 15.0% | 526 | 14.7% | 311 | 215 |

## Classification

ROC-AUC/PR-AUC are threshold-independent (rank-based) and unaffected by the cutoff below. Sensitivity/specificity/balanced accuracy ARE threshold-dependent and are reported at TWO cutoffs so the effect of proper threshold selection is visible rather than hidden: the untuned default (0.5) that earlier reports used, and the operating threshold selected on the VALIDATION set only (see `select_operating_thresholds.py`).

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier | Sens. @0.5 | Spec. @0.5 | Operating thr. | Sens. @thr. | Spec. @thr. | Bal.Acc. @thr. | Thr. selected on | Confusion matrix |
|---|---|---|---|---|---|---|---|---|---|---|---|
| whole_image_baseline | 0.688 [0.635, 0.735] | 0.660 [0.601, 0.721] | 0.240 | 0.744 | 0.479 | 0.408 | 0.879 | 0.257 | 0.568 | val (target 90%) | `phase14_confusion_matrix_whole_image_baseline.png` |
| lesion_crop | 0.799 [0.764, 0.838] | 0.737 [0.679, 0.806] | 0.183 | 0.690 | 0.742 | 0.189 | 0.911 | 0.418 | 0.664 | val (target 90%) | `phase14_confusion_matrix_lesion_crop.png` |
| multimodal | 0.756 [0.713, 0.800] | 0.714 [0.655, 0.772] | 0.207 | 0.591 | 0.723 | 0.181 | 0.870 | 0.492 | 0.681 | val (target 90%) | `phase14_confusion_matrix_multimodal.png` |
| ensemble_baseline_multimodal | 0.760 [0.715, 0.801] | 0.725 [0.669, 0.780] | 0.198 | 0.642 | 0.723 | 0.358 | 0.837 | 0.466 | 0.652 | val (target 90%) | `phase14_confusion_matrix_ensemble.png` |

Confusion matrices are saved as PNG figures in `reports/figures/` (filenames listed in the table above), each annotated with both raw counts and row-normalized percentages at the model's operating threshold.

The `ensemble` row (baseline + multimodal, probabilities averaged) is a standard accuracy-improvement technique -- compare its ROC-AUC/CI directly against baseline and multimodal alone to see whether averaging genuinely helped or just added noise on this test set.

## Localization

- map: 0.0623
- map_50: 0.1627
- map_75: 0.0249
- mean_top1_iou: 0.2312

## Segmentation

- dice: 0.2548
- iou: 0.1870

## Explainability (qualitative + overlap)

Mean fraction of ground-truth lesion box covered by high-attention malignant-class Grad-CAM pixels: 7.56%  95% CI [5.10%, 10.20%]  (n=200)

Grad-CAM is generated for the malignant-class score specifically, regardless of the model's own top prediction on a given image, so this number consistently answers "where does the model see evidence of malignancy" rather than mixing that question with "evidence for whatever this model happened to predict."
See figure: C:\Users\21ayu\Desktop\memo\mammotwin\reports/figures\phase14_explainability_test_examples.png

## Robustness

Not evaluated — no external dataset was available for this project. Stated here as an explicit limitation.
