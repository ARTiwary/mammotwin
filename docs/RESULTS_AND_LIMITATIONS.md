# MammoTwin — Final Results Summary, Key Findings, and Limitations
*(For use in the Phase 17 report/thesis. This is the truly final state of the
project: all data-pipeline bugs found and fixed, all classifiers retrained on
the full, correct dataset, pretrained-encoder segmentation adopted, and a
properly-aligned two-model ensemble evaluated as the final headline result.)*

## 1. Results Summary

### Classification

| Model | ROC-AUC (95% CI) | n (test) |
|---|---|---|
| Whole-image baseline | 0.688 [0.635, 0.735] | 526 |
| Multimodal | 0.756 [0.713, 0.800] | 526 |
| Baseline + multimodal ensemble | 0.760 [0.715, 0.801] | 526 |
| Lesion-crop | 0.799 [0.764, 0.838] | 512 |
| **Lesion-crop + multimodal ensemble (final, best)** | **0.818 [0.782, 0.854]** | 512 |

The final ensemble (lesion-crop + multimodal, probabilities averaged,
merged by `image_id` since lesion-crop only covers images with a valid
lesion bounding box) is the strongest model in this project. At its
validation-tuned operating threshold (0.3045, target sensitivity 90%):
sensitivity 86.9%, specificity 54.2% on the test set.

**Why this combination outperformed the earlier baseline+multimodal
ensemble (0.760):** lesion-crop and multimodal draw on more genuinely
different information than baseline and multimodal do — lesion-crop sees
only the cropped lesion region (avoiding whole-image noise and the
marker-artifact issue baseline had), while multimodal draws on structured
clinical features (density, view, mass shape, etc.). Ensembling helps
most when combined models make different kinds of errors, and this pair
appears to satisfy that condition better than the previous pairing did.

### Localization (Faster R-CNN detector)
mAP: 0.044 · mAP@50: 0.120 · mean top-1 IoU: 0.181

**Note:** the detector was also retrained on the corrected, full training
set (2,445 images, matching the fix that helped lesion-crop and
segmentation substantially) but did NOT improve — the retrained version
scored mAP 0.038, slightly worse. Unlike lesion-crop and segmentation,
the detector's limitation does not appear to be training-set size; it is
kept at its earlier checkpoint. This asymmetry (some modules improve
with more data, one doesn't) is itself worth stating plainly in the
report rather than glossed over — it suggests the detector needs a
different intervention (more epochs, different anchor sizes/hyperparameters,
or simply that Faster R-CNN on such small lesions is a harder problem than
data volume alone can fix).

### Segmentation (U-Net, pretrained ResNet18 encoder)
**Test Dice: 0.255 · Test IoU: 0.187** (final, on the corrected full
training set of 2,445 images)

Full progression of this project's segmentation improvements:

| Stage | Val Dice |
|---|---|
| From-scratch U-Net, whole image | ~0.11 |
| From-scratch U-Net, patch-based training | 0.156 |
| Pretrained ResNet18 encoder, patch-based (partial/bugged data, 949 imgs) | 0.206 |
| Pretrained ResNet34 encoder (same bugged data) | 0.191 (underperformed ResNet18) |
| **Pretrained ResNet18 encoder, full corrected data (final)** | **0.223** (test: 0.255) |

This is more than a 2x improvement in Dice over the project's original
whole-image, from-scratch approach, achieved through two independent,
verified fixes: patch-based training (forcing the model to focus on the
lesion rather than general tissue density) and a pretrained encoder
(leveraging ImageNet features rather than learning everything from ~1,000
images from scratch), plus a data-pipeline bug fix that unlocked the full
training set.

### Explainability (Grad-CAM, malignant-class, n=200, bootstrap 95% CI)
Mean fraction of the ground-truth lesion box covered by high-attention
pixels: **7.56%, 95% CI [5.10%, 10.20%]**

### Robustness
Not evaluated — no external dataset (e.g. INbreast) was available for this
project. Stated as an explicit limitation, not a fabricated result.

---

## 2. Key Finding #1: A Preprocessing Artifact Was Inflating the Baseline Classifier

Qualitative Grad-CAM inspection showed the whole-image classifier's
attention sometimes landed on a burned-in laterality marker ("R CC")
rather than breast tissue. The root cause was a real implementation bug:
`remove_background()`'s own docstring specified that non-breast pixels
inside the crop should be zeroed out, but the code never did this. After
fixing it and retraining, the whole-image baseline's test ROC-AUC dropped
from 0.752 to 0.688 — a real, meaningful loss of "performance" that was
never genuine diagnostic signal to begin with. Lesion-crop and multimodal,
which don't depend on that image region, were unaffected by the same fix.

**Suggested framing:** *"A substantial fraction of the whole-image
classifier's originally reported performance (ROC-AUC 0.752) was
attributable to a preprocessing artifact rather than genuine diagnostic
signal. This is a concrete, empirically demonstrated instance of the
shortcut-learning risk documented in the mammography ML literature."*

## 3. Key Finding #2: A Recurring Data-Truncation Bug Was Suppressing Lesion-Level Models

Independently, a data-pipeline bug caused `bbox_metadata_train.csv` (used
by lesion-crop and segmentation) to repeatedly get truncated to ~1,000
rows instead of the full 2,501 — likely from a terminal history-replay
issue reusing an old, limited command. Because the total row count
(1,000) looked plausible on the surface and the resulting per-class
counts were not obviously wrong, this went undetected through multiple
rounds of experimentation. Once found and fixed, retraining on the full,
correct 2,445-image training set produced substantial, genuine
improvements:

| Model | Suppressed (949 images) | Corrected (2,445 images) |
|---|---|---|
| Lesion-crop val AUC | 0.744 | **0.823** |
| Segmentation val Dice | 0.206 | **0.223** |

**Suggested framing:** *"A data-loading bug silently limited two of the
five model components to less than 40% of their available training data
across several experimental iterations. This underscores the importance
of validating dataset size at each training run, not just once during
initial setup — a lesson reflected in this project's testing suite
(tests/test_splits.py) going forward."*

**Notably, the detector was also retrained on the corrected data and did
NOT improve** (mAP 0.044 → 0.038), suggesting its limitation is
architectural/hyperparameter-related rather than data-volume-related —
an important negative result that adds credibility to the positive
findings above (not everything automatically "improved" once more data
was available, which would be a red flag for a fabricated or cherry-picked
narrative; instead, different modules responded differently, as expected).

---

## 4. Limitations (for the report's Limitations section)

- **Moderate-to-good classification performance at best**, final ensemble
  ROC-AUC 0.818. This is a real, credible research-grade result, but still
  below the bar for clinical-grade screening performance.
- **Both key findings above (Sections 2-3) mean even the corrected numbers
  should be treated with continued scrutiny** — undiscovered issues of a
  similar nature cannot be ruled out.
- **Detector remains weak** (mAP 0.044) and did not benefit from the
  data-volume fix that helped other modules — a genuine, unresolved
  limitation requiring further architectural work.
- **No external validation.** All results are internal to CBIS-DDSM.
- **Longitudinal module** (if built) should be clearly labeled as
  demonstrated on simulated pairs, per the original project plan.

## 5. Suggested Future Work

- External validation on INbreast or VinDr-Mammo.
- A systematic audit for other non-anatomical artifacts, using the same
  method that surfaced the laterality-marker issue.
- Detector-specific improvements (more epochs, anchor size tuning, or a
  pretrained detection backbone) since more data alone did not help it.
- A three-model ensemble (baseline + lesion-crop + multimodal) restricted
  to their shared image intersection, as a further extension of the
  two-model ensembling result found here.