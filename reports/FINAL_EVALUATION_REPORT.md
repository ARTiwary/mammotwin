# MammoTwin — Phase 14 Final Evaluation Report

Generated: 2026-09-13T21:44:36

Locked test set: 526 rows, 235 patients. This test set was untouched by any training or model-selection decision prior to this evaluation.

## Classification

ROC-AUC/PR-AUC are threshold-independent (rank-based) and unaffected by the cutoff below. Sensitivity/specificity/balanced accuracy ARE threshold-dependent and are reported at TWO cutoffs so the effect of proper threshold selection is visible rather than hidden: the untuned default (0.5) that earlier reports used, and the operating threshold selected on the VALIDATION set only (see `select_operating_thresholds.py`).

| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier | Sens. @0.5 | Spec. @0.5 | Operating thr. | Sens. @thr. | Spec. @thr. | Bal.Acc. @thr. | Thr. selected on |
|---|---|---|---|---|---|---|---|---|---|---|
| whole_image_baseline | 0.659 [0.607, 0.705] | 0.626 [0.564, 0.685] | 0.257 | 0.642 | 0.582 | 0.284 | 0.828 | 0.286 | 0.557 | val (target 90%) |
| lesion_crop | 0.730 [0.686, 0.774] | 0.633 [0.561, 0.713] | 0.221 | 0.563 | 0.742 | 0.160 | 0.906 | 0.341 | 0.624 | val (target 90%) |
| multimodal | 0.756 [0.713, 0.800] | 0.714 [0.655, 0.772] | 0.207 | 0.591 | 0.723 | 0.181 | 0.870 | 0.492 | 0.681 | val (target 90%) |

## Explainability (qualitative + overlap)

Mean fraction of ground-truth lesion box covered by high-attention malignant-class Grad-CAM pixels: 8.26%  95% CI [5.68%, 11.06%]  (n=200)

Grad-CAM is generated for the malignant-class score specifically, regardless of the model's own top prediction on a given image, so this number consistently answers "where does the model see evidence of malignancy" rather than mixing that question with "evidence for whatever this model happened to predict."
See figure: C:\Users\21ayu\Desktop\memo\mammotwin\reports/figures\phase14_explainability_test_examples.png

## Robustness

Not evaluated — no external dataset was available for this project. Stated here as an explicit limitation.
