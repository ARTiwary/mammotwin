# Phase 8 — Lesion Segmentation

## Prerequisite

Uses `bbox_metadata_train.csv` / `bbox_metadata_val.csv` from Phase 7 —
they already contain the resolved mask path per row (`mask_path_used`), so
no new ground-truth-building step is needed. If you haven't run Phase 7's
`build_bbox_metadata.py` for both splits yet, do that first.

## Run it

```powershell
python scripts\run_phase8_segmentation.py --epochs 20
```

Like Phase 7, segmentation is resolution-sensitive for small lesions —
override independently of the classifier's config if needed:
```powershell
python scripts\run_phase8_segmentation.py --epochs 20 --image-size 384
```

Produces:
- `models\segmentation_unet_<timestamp>.pt` — best checkpoint (by val Dice)
- `reports\figures\phase8_segmentation_examples.png` — input / ground-truth
  mask / predicted mask overlays for a few validation examples
- Printed Dice and IoU per epoch

## A real bug I caught while testing — worth understanding

Lesion pixels are typically under 1% of a mammogram. Plain BCE loss lets a
segmentation model achieve deceptively low, steadily-falling loss just by
predicting "background everywhere" — every pixel probability can sit below
the 0.5 threshold indefinitely, which means **Dice/IoU can get stuck at
exactly 0.0 even while the loss curve looks like it's improving**. I hit
this myself during testing (15 epochs, loss fell from 0.82 to 0.69, Dice
stayed at 0.0000 the entire time) and fixed it with dynamically-computed
positive-class weighting inside the BCE component (`DiceBCELoss` in
`segmentation_metrics.py`) — the same inverse-frequency idea as Phase 6's
class weighting, applied per-pixel instead of per-image. After the fix, the
same setup reached Dice 0.545 in 15 epochs on synthetic data.

**Watch for this exact pattern in your own real-data run**: if `val_dice`
stays at 0.0000 for many epochs while `val_loss` is clearly decreasing,
something is still off (e.g. a mask/image alignment bug) — don't assume
"just needs more epochs" the way I almost did.

## What I verified vs. what needs your real data

**Verified, with an actual before/after fix on a real bug:**
- The BCE class-imbalance trap above
- Mask/image spatial alignment through Phase 4's background-removal crop
  (same coordinate-transform logic validated in Phase 7)
- Full training loop, checkpointing, Dice/IoU computation, overlay
  visualization all run end-to-end

**Not verified — needs your real data:**
- Actual Dice/IoU on real lesions (meaningless on synthetic noise images)
- Whether `bce_weight: 0.5` in `config.yaml`'s new `segmentation:` section
  is the right balance for real mammogram lesion sizes — if Dice is slow to
  get going on real data, try lowering it (e.g. 0.3) to lean more on Dice
  loss, which is inherently more robust to class imbalance than BCE

## Honest expectation-setting

Like Phase 7, this is a genuinely hard problem with limited data (however
many masks resolved successfully in your Phase 7 run — check that number
again here, since segmentation needs the same masks). Published mammography
segmentation papers with thousands of images and specialized architectures
report Dice scores often in the 0.6–0.8 range; a smaller from-scratch U-Net
on a few hundred/thousand images may land lower. That's expected and still
a legitimate, reportable result — document it as a first-pass segmentation
baseline with a clearly hard, small dataset, similar to how Phase 7's
localization results were framed.