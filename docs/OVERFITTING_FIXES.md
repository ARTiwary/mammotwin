# Fixing the Overfitting Problem (post-Phase-14 retrofit)

Phase 14's locked-test-set results confirmed what the training curves had
been hinting at since Phase 6: the whole-image and multimodal classifiers
overfit meaningfully during training (val AUC 0.79 on validation dropped
to 0.68 on the untouched test set). Three concrete fixes — each flagged as
a known gap earlier in this project but never actually wired in — are
applied here, across every classifier (Phases 6, 9, 13) and the
segmentation model (Phase 8).

## 1. Data augmentation (biggest expected impact)

New: `src/preprocessing/augmentation.py` — random horizontal flip, small
rotation (±15°), and brightness/contrast jitter, implemented with only
numpy/cv2 (no Albumentations — it pulls in scipy transitively, the same
blocked-DLL risk this project has already hit twice).

- `augment_image()` for the classifiers (Phase 6, 9, 13).
- `augment_pair()` for segmentation (Phase 8) — applies the SAME random
  flip/rotation to both image and mask so they stay geometrically
  synchronized. Verified directly: after augmentation, a mask and its
  corresponding bright image region still overlap with 0.986 IoU.

Augmentation is applied **only to training data, never validation or
test** — every dataset class now takes an `augment: bool` parameter, and
every training script explicitly sets `augment=False` for its val/test
datasets. Confirmed empirically that it's actually doing something: two
reads of the same training image differ when `augment=True`, and are
byte-identical when `augment=False`.

Disable it if needed: `--no-augment` on any of the four training scripts.

## 2. Excluding Phase 4 quality-flagged images from training

New: `src/data/quality_filter.py`. Every classifier/segmentation training
script now automatically drops training rows that Phase 4's
`quality_report.csv` flagged (`passed == False`) — blank images,
low-contrast images, failed background segmentation. This was called out
as a known gap in Phase 6's very first delivery and never actually
implemented until now.

Important design choice, tested directly: an image_id **absent** from the
quality report (never checked) is KEPT, not excluded — the filter only
acts on positive evidence of a problem, never assumes badness from
missing data. Validation and test sets are deliberately NOT filtered —
a model should be evaluated on the real distribution it will actually
face, quality issues included.

Disable it if needed: `--no-quality-filter`. Point at a specific report
with `--quality-report path/to/quality_report.csv` if yours isn't at the
default location.

## 3. Learning-rate scheduling

Added `torch.optim.lr_scheduler.ReduceLROnPlateau` (factor=0.5,
patience=2) to all four training loops (Phase 6, 8, 9, 13). When
validation loss stops improving for 2 consecutive epochs, the learning
rate is halved — directly targeting the exact pattern seen in every
training run so far (train loss still falling while val loss climbs):
a high learning rate late in training pushes the model to keep fitting
training-set noise. You'll see a printed message whenever this triggers:
`-> Reducing learning rate: 1.00e-04 -> 5.00e-05`.

## 4. A smaller change to segmentation's loss balance

`config.yaml`'s `segmentation.bce_weight` default lowered from 0.5 to
0.3 — leaning more on Dice loss, which handles the severe
foreground/background pixel imbalance in lesion masks more robustly than
BCE. This was recommended but not applied back in Phase 8.

## What to do now

Re-run training for whichever models you want to improve — all four
scripts work exactly as before, just with these fixes on by default:

```powershell
python scripts\run_phase6_baseline.py --num-workers 4
python scripts\run_phase9_lesion_crop.py
python scripts\run_phase13_multimodal.py
python scripts\run_phase8_segmentation.py --epochs 20
```

**Important**: once you retrain any model, its OLD Phase 14 test-set
result is no longer the right one to report for that model — re-run
Phase 14 once more, after all retraining is done, for a single, final,
genuinely-final set of numbers. Don't evaluate on the test set after
every individual retraining round; batch your changes, retrain
everything you want to fix, and touch the test set exactly once more.

## What I verified before sending this

- Augmentation actually varies the image (confirmed: two augmented reads
  of the same image differ; two non-augmented reads are identical).
- Paired image+mask augmentation stays synchronized (0.986 IoU test).
- The quality filter correctly excludes flagged images, keeps unflagged
  and unreported ones (a small hand-built test case with known expected
  output).
- All four modified training scripts (6, 8, 9, 13) still run end-to-end
  without errors in demo mode after these changes.

## What I could NOT verify from here

Whether these fixes actually close the validation-to-test AUC gap on
your REAL data — that requires your real training run. The mechanisms
here are standard, well-established techniques for exactly this failure
pattern, but "should help" isn't the same as "verified to help" until you
see the real before/after numbers.