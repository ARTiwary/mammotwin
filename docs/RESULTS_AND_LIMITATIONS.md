# MammoTwin — Results Summary, Key Finding, and Limitations
*(For use in the Phase 17 report/thesis. Numbers below are from the corrected,
final Phase 14 run on the locked test set — 526 rows, 235 patients — using
the specific checkpoints verified in `models/registry.json` after the
sensitivity-threshold, segmentation, and explainability fixes documented in
`docs/SENSITIVITY_THRESHOLD_FIX.md`.)*

## 1. Results Summary

### Classification

| Model | ROC-AUC (95% CI) | Operating threshold* | Sensitivity | Specificity |
|---|---|---|---|---|
| Whole-image baseline | 0.752 [0.713, 0.796] | 0.160 | 0.874 | 0.450 |
| Lesion-crop | 0.726 [0.680, 0.769] | 0.048 | 0.916 | 0.281 |
| Multimodal | 0.758 [0.718, 0.799] | 0.053 | 0.916 | 0.318 |

\*Selected on the validation set only, targeting ≥90% sensitivity (see
`select_operating_thresholds.py`). At the untuned default of 0.5, all three
models measured only 52–70% sensitivity — the tuned threshold is what makes
these numbers usable for a screening-oriented research prototype at all.

**Interpretation:** moderate, roughly-equal discriminative ability across
all three model variants (ROC-AUC 0.73–0.76, overlapping confidence
intervals — none is clearly superior to the others). Pushing sensitivity to
~90% costs substantial specificity (28–45%), which is the direct, expected
consequence of a moderate AUC rather than a flaw in the thresholding
procedure itself: no threshold choice can buy sensitivity and specificity
simultaneously when the underlying ranking ability is this limited.

### Localization (Faster R-CNN detector)
mAP: 0.058 · mAP@50: 0.153 · mAP@75: 0.027 · mean top-1 IoU: 0.207

### Segmentation (patch-based U-Net)
Dice: 0.161 · IoU: 0.104

### Explainability (Grad-CAM, malignant-class, n=200, bootstrap 95% CI)
Mean fraction of the ground-truth lesion box covered by high-attention
pixels: **4.98%, 95% CI [3.09%, 7.17%]**

### Robustness
Not evaluated — no external dataset (e.g. INbreast) was available for this
project. Stated as an explicit limitation, not a fabricated result.

---

## 2. Key Finding: Evidence of Shortcut Learning

This is the most important qualitative result from Phase 14 and deserves
its own subsection in the report, not just a line in a limitations table.

The quantitative Grad-CAM overlap (4.98%) already indicates that the
classifier's malignant-class evidence rarely coincides with the annotated
lesion. Visual inspection of the qualitative examples
(`phase14_explainability_test_examples.png`) suggests a specific reason
why: in at least one example, the model's highest-attention region falls
directly on the **laterality/view marker text burned into the corner of
the image** (e.g. "R CC"), rather than on breast tissue.

This is a documented failure mode in mammography machine learning research
— CBIS-DDSM (like most mammography datasets) embeds acquisition metadata
directly in image pixels, and a discriminative classifier is free to use
any pixel that correlates with the label, including markers that
correlate with laterality, acquisition site, or scanner defaults rather
than pathology. A model can achieve a real, non-trivial AUC (0.75) partly
through such shortcuts without the AUC number itself revealing this.

**Suggested framing for the report:**

> Quantitative overlap analysis (4.98%, 95% CI [3.1%, 7.2%], n=200)
> confirms the classifier's saliency maps frequently do not localize to
> the annotated lesion. Qualitative inspection suggests partial reliance
> on non-anatomical image artifacts (e.g., laterality markers), a
> known shortcut-learning risk in mammography classification. This
> reinforces that Grad-CAM overlays in this system must be treated as
> directional evidence of the model's attention, not a localization or
> diagnostic signal, and that the classifier's moderate ROC-AUC should
> not be read as evidence that it is reasoning about lesions the way a
> radiologist would.

**A note on rigor, for the methods section:** this finding itself
illustrates why the project's evaluation infrastructure needed fixing
before it was trustworthy — the original overlap estimate (13.97%, n=6)
had a bootstrap-estimated true range of roughly 6–28%, too imprecise to
support any claim at all. It was only once the sample size was increased
to 200 and the metric was redefined to consistently probe the
malignant-class score (rather than mixing in benign-predicted images'
irrelevant heatmaps) that a statistically defensible, and actionable,
result emerged.

---

## 3. Limitations (for the report's Limitations section)

- **Moderate, not strong, classification performance.** ROC-AUC of
  0.73–0.76 is a real but modest discriminative signal. The high
  sensitivity operating point (~90%) is only achievable at the cost of
  flagging 55–70% of benign cases for review — appropriate for a research
  prototype's safety-first design, but not competitive with a clinical
  screening tool.
- **Evidence of shortcut learning** (Section 2) means the classifier's
  decisions cannot currently be trusted to reflect lesion-focused
  reasoning, even where its predictions are correct.
- **Weak localization and segmentation.** mAP (0.058) and Dice (0.161)
  indicate the detector and segmentation model have learned only a
  coarse, unreliable sense of lesion location — attributable to the
  small training set (~2,400 images) and from-scratch (non-pretrained)
  architectures, not to a discovered implementation defect (dataset
  alignment, loss weighting, and augmentation correctness were each
  specifically verified during debugging).
- **No external validation.** All results are internal to a single
  train/val/test split of one dataset (CBIS-DDSM); generalization to
  other scanners, populations, or acquisition protocols is untested.
- **Longitudinal module** (if built) should be clearly labeled as
  demonstrated on simulated pairs unless genuine prior/current
  examinations were used, per the original project plan.

## 4. Suggested Future Work

- Retrain with the acquisition-marker regions masked or cropped out
  before training, and re-measure both AUC and Grad-CAM overlap — if AUC
  drops substantially, that would be direct confirmation of how much of
  the current performance is shortcut-driven.
- A pretrained encoder backbone (e.g., ImageNet-pretrained ResNet/U-Net
  encoder) for segmentation, which the literature consistently shows
  helps most on small medical-imaging datasets like this one.
- External validation on INbreast or VinDr-Mammo, per the original
  dataset strategy, to test whether the shortcut-learning finding and
  the AUC estimate hold outside CBIS-DDSM.
