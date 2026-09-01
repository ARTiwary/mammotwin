# Phase 14 — Final Evaluation (THE test set, touched once)

## Before you run this — read this section

This is the single most consequential script in the project. It evaluates
every trained model against `test_split.csv` — the patient-level split
that has been **locked since Phase 3** and never touched by any training
run or hyperparameter decision. Per the plan's golden rule: choose/tune
models using validation data only; touch the test set once, here, for the
final unbiased estimate.

**Do not re-run this repeatedly while tuning anything.** If you find
yourself wanting to run this again after seeing a disappointing number and
going back to adjust something, that defeats the entire purpose of having
a locked test set — you'd be tuning on the test set by proxy. If you
genuinely need to make further changes to a model, that's fine, but the
correct move afterward is to treat the NEW result as the one true final
number, not to average/cherry-pick across multiple test-set runs.

## Run it

```powershell
python scripts\run_phase14_final_evaluation.py --i-understand-this-locks-the-test-set --raw-images-dir "C:\Users\21ayu\Desktop\memo\mammotwin\Datasets\jpeg" --detector-checkpoint models\detector_fasterrcnn_<timestamp>.pt --segmentation-checkpoint models\segmentation_unet_<timestamp>.pt
```

The `--i-understand-this-locks-the-test-set` flag is **required** —
deliberately, so this can't be run by accident or muscle memory.

**Checkpoint discovery**: the whole-image baseline (Phase 6), lesion-crop
(Phase 9), and multimodal (Phase 13) classifiers are auto-discovered from
`models\registry.json` (best `val_auc` per phase) — no need to specify
them unless you want to override. The detector (Phase 7) and segmentation
model (Phase 8) were never logged to the registry by their scripts, so
you must pass `--detector-checkpoint` and `--segmentation-checkpoint`
explicitly (check `models\` for the filenames, or skip either with
`--skip-localization` / `--skip-segmentation` if you don't want to
evaluate them here).

## What it evaluates, matching the plan's Phase 14 checklist exactly

- **Classification** (all 3 classifiers): ROC-AUC, PR-AUC, sensitivity,
  specificity, precision, F1, balanced accuracy — **plus bootstrap 95%
  confidence intervals** on ROC-AUC and PR-AUC (1000 resamples by default).
- **Calibration**: reliability diagram + Brier score, per classifier.
- **Localization**: IoU/mAP on the test set, using the same custom
  torchvision-only metrics from Phase 7 (no sklearn/torchmetrics needed).
- **Segmentation**: Dice and IoU on the test set.
- **Explainability**: a qualitative Grad-CAM figure on test examples, plus
  a simple quantitative measure — what fraction of each ground-truth
  lesion box is covered by the heatmap's highest-attention pixels.
- **Robustness**: explicitly reported as **skipped**, not silently
  omitted — no external dataset (e.g. INbreast) was downloaded for this
  project. State this as a genuine limitation in your report; don't try
  to paper over it.

Everything gets written to `reports\FINAL_EVALUATION_REPORT.md` — a
ready-to-adapt Results-section table, plus individual calibration figures
per model and the explainability figure, all in `reports\figures\`.

## I tested this extremely thoroughly before sending it

Given how much rides on this script working correctly on the first real
run, I built a complete synthetic test scenario — fake but realistic
checkpoints for all 5 model types (classifier ×3, detector, segmentation),
a populated `registry.json`, and a fake locked test set with real
mask-derived bounding boxes — and ran the **actual script** against it,
not just individual functions in isolation. Confirmed:
- All three classifiers load, run inference, and produce bootstrapped
  CIs correctly.
- The `bbox_metadata_test.csv` auto-build path works (tested by running
  once with the file present, once with it deleted, forcing an on-the-fly
  build — both succeeded, 20/20 boxes extracted in the from-scratch case).
- Localization, segmentation, and explainability evaluation all completed
  without errors and produced correctly-shaped, correctly-rendered output
  (checked the calibration figures, the report's markdown table, and the
  explainability overlay figure directly).

The numbers from that test run are meaningless (random-weight checkpoints,
20 fake images) — what's proven is that the *pipeline* is correct, so your
real numbers should come out right on the first genuine run.

## One judgment call worth knowing about

The lesion-crop classifier's test evaluation uses `bbox_metadata_test.csv`
rather than the plain `test_split.csv` — meaning its effective test set is
whatever subset of test images successfully yielded a usable bounding box
(the same `mask_shape_mismatch`-style attrition seen in Phase 7). This
means the three classifiers are technically evaluated on slightly
different-sized test subsets. Worth a one-line disclosure in your report's
methods section rather than treating all three numbers as perfectly
apples-to-apples.