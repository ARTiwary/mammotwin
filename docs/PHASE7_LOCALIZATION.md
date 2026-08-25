# Phase 7 — Lesion Localization

## Background: why this phase needed extra care

CBIS-DDSM doesn't ship simple (x, y, w, h) bounding boxes — it ships binary
ROI **mask** images, confirmed (via the dataset's official paper) to be the
same pixel dimensions as their full mammogram. A bounding box is derived by
finding the extent of the mask's foreground pixels.

There's a known wrinkle: the "cropped image" and "ROI mask" files sometimes
share the same series UID folder (the same ambiguity you hit in Phase 3),
so a folder can contain 2 files with no way to tell from the filename which
is the mask and which is the small cropped patch. This pipeline
disambiguates them by comparing pixel dimensions: the file close to the
full mammogram's size is the mask, the much smaller one is the crop.

I tested the whole chain — mask/crop disambiguation, bounding-box
extraction, and the crop+resize coordinate transform needed to keep the box
aligned after Phase 4's background-removal preprocessing — against a
synthetic case with a KNOWN correct answer, and it matched exactly.

## Step 1 — Build bounding-box ground truth

```powershell
python scripts\build_bbox_metadata.py --raw-images-dir "C:\Users\21ayu\Desktop\memo\mammotwin\Datasets\jpeg" --limit 300
```

Start with `--limit 300` for a first pass (this loads full-resolution
images to check shapes, which is slower than earlier scripts) — raise it
once you've confirmed the output looks right. Produces
`data\metadata\bbox_metadata.csv` with one row per finding: pixel + normalized
bbox coordinates, a `fill_ratio` quality signal, and an `ambiguous_roi_folder`
flag worth spot-checking if it's ever True.

**Watch the "Failure reasons breakdown" output.** Some rows won't yield a
usable box — most commonly `mask_shape_mismatch` (the mask's dimensions
don't match the full image, which can happen with real-world dataset
inconsistencies) or `no_mask_candidates_found` (path resolution issue,
similar to what we debugged in Phase 3). A moderate failure rate is
expected and fine; if it's the majority of rows, share the breakdown and
we'll dig in.

## Step 2 — Train and evaluate the detector

```powershell
python scripts\run_phase7_localization.py --epochs 10
```

Trains a Faster R-CNN (single class: "lesion" vs. background) on the boxes
from Step 1. Produces:
- `models\detector_fasterrcnn_<timestamp>.pt`
- Printed IoU/mAP metrics (map, map_50, map_75, mar_1, mar_10, mar_100, etc.)
- `reports\figures\phase7_predicted_boxes.png` — a few validation examples
  with ground-truth boxes in green, predictions in red

Faster R-CNN is noticeably heavier than the Phase 6 classifier — expect
slower training, especially on CPU. If you hit memory errors, drop
`--batch-size` to 2 or 1:
```powershell
python scripts\run_phase7_localization.py --epochs 10 --batch-size 2
```

## What I could and couldn't verify from here

**Verified, with a known-correct synthetic test case:**
- Mask vs. cropped-patch disambiguation by size
- Bounding box extraction from a mask (exact pixel match)
- The coordinate transform that carries a box through Phase 4's
  background-removal crop + resize

**Verified, pipeline runs end-to-end without crashing:**
- Full training loop (loss computation, backprop, checkpointing)
- mAP evaluation via torchmetrics
- Prediction visualization

**NOT verified — needs your real data:**
- Actual mAP/IoU numbers on real lesions (meaningless on synthetic noise
  images, which is all I have access to)
- How often `mask_shape_mismatch` or other failures occur on the real
  CBIS-DDSM masks — this is worth checking as your very first step, since
  it determines how much usable localization data you actually have

## A caught bug worth knowing about

While testing, I found that `weights=None` alone does NOT stop torchvision
from downloading an ImageNet-pretrained ResNet50 **backbone** for Faster
R-CNN — there's a separate `weights_backbone` parameter that defaults to
pretrained regardless. Already fixed in `detector.py`, but worth knowing
if you ever see an unexpected download when you expected a from-scratch
model.