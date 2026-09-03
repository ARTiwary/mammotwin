# Fix: Low Sensitivity in Phase 14 Classification Results

## The problem

Phase 14's final evaluation reported sensitivity of ~52-53% for the
whole-image and multimodal classifiers. On a malignant/benign task, that
means the models were missing roughly **half of the malignant cases** —
clearly not acceptable, even for a research prototype whose stated purpose
is to flag suspicious cases for expert review.

## Root cause

This was not a model-quality problem — it was a **threshold-selection
bug**. `compute_classification_metrics()` (`src/utils/metrics.py`) accepts
a `threshold` argument but every call site in the project (Phase 6, 9, 13
training scripts and the original Phase 14 script) called it with no
argument, silently falling back to `threshold=0.5`.

0.5 is not a meaningful cutoff for this task:

- The dataset is imbalanced (more benign than malignant cases), so a raw
  0.5 cutoff on softmax output is biased toward predicting the majority
  class.
- Even on a balanced dataset, 0.5 encodes the assumption that a false
  negative (missed cancer) and a false positive (unnecessary review) are
  equally costly. They are not — this project's entire uncertainty/review
  design (Phase 11) exists specifically because that assumption is wrong.

No amount of retraining fixes a threshold problem, because ROC-AUC (which
was already reasonable, ~0.75-0.82) measures the *ranking* of predictions,
independent of where the cutoff is drawn. The fix is to choose the cutoff
correctly, not to retrain.

## The fix

1. `src/utils/stats.py` gained three new functions: `roc_points`,
   `select_threshold_for_target_sensitivity`, and
   `youden_optimal_threshold` — all pure numpy, consistent with the rest
   of the file, with self-tests in the `__main__` block.
2. `config/config.yaml` gained `training.target_sensitivity: 0.90` — the
   minimum sensitivity every classifier's operating threshold must try to
   meet. (90% is a reasonable placeholder for a *research* system; a
   deployed screening tool would set this per clinical guidance, not from
   a config file default.)
3. New script `scripts/select_operating_thresholds.py` loads each trained
   classifier, runs it on the **validation split only**, and picks the
   most specific threshold that still reaches the target sensitivity. It
   saves the result to `data/metadata/operating_thresholds.json` along
   with a sensitivity/specificity-vs-threshold plot per model.
4. `scripts/run_phase14_final_evaluation.py` now loads that file and
   reports classification metrics at **two** cutoffs, side by side: the
   untuned 0.5 (kept only as a reference point) and the tuned operating
   threshold. Nothing is hidden — if a model can't reach the target
   sensitivity even on validation data, the report says so explicitly
   instead of silently reporting whatever the tuned number happens to be.

## Why this doesn't violate the "touch the test set once" rule

The threshold is selected entirely from the **validation** split, before
Phase 14 ever runs. Phase 14 only *applies* an already-chosen threshold to
the test set — it does not search for one there. This is the same
principle the project already applies to model selection (Phase 6/9/13
pick the best checkpoint using validation AUC; the test set only scores
the final choice).

## How to apply this fix

```bash
# 1. Choose thresholds from the validation set (safe to re-run any time)
python scripts/select_operating_thresholds.py

# 2. Re-run the final evaluation — it will pick up operating_thresholds.json automatically
python scripts/run_phase14_final_evaluation.py --i-understand-this-locks-the-test-set \
    --raw-images-dir "<path-to-your-CBIS-DDSM-jpeg-dir>"
```

## What this fix does NOT solve

- It does not fix the model's underlying discriminative ability — if
  ROC-AUC is mediocre, a better threshold trades sensitivity for
  specificity but can't manufacture information the model doesn't have.
- It does not replace **calibration** (Phase 9's Brier score / reliability
  diagrams already cover that) — calibration affects how trustworthy the
  probability values are; thresholding only decides where to cut them.
- If a model cannot reach `target_sensitivity` even on validation data
  (the script will warn about this explicitly), that is a genuine
  model-quality limitation to report honestly, not something a threshold
  choice can paper over.
