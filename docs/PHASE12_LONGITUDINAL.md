# Phase 12 — MammoTwin Longitudinal Module

## The most important thing to understand about this phase

**CBIS-DDSM has no genuine prior/current exam pairs for the same patient**
(established back in Phase 1). Every timeline this script produces is
**SIMULATED**: real breast anatomy from one actual exam, with a synthetic
lesion trajectory (absent → partial → full, matching the real Phase 7
mask) layered on top. The script prints a bright red disclaimer banner on
every figure and every run, and it should never be described, shown, or
quoted as if it demonstrates real disease progression. It demonstrates the
**software's** ability to register, compare, and quantify change between
two mammograms — nothing more, and the plan is explicit that this is the
correct, honest way to handle this phase given the dataset available.

## Run it

```powershell
python scripts\run_phase12_longitudinal.py --checkpoint models\baseline_resnet50_20260825_114630.pt --n-patients 3
```

Or without a classifier (skips model-score-change reporting, only
area/overlap changes):
```powershell
python scripts\run_phase12_longitudinal.py --n-patients 3
```

## What it produces

For each patient: a 3-panel timeline figure
(`reports\figures\phase12_timeline_<i>.png`) — baseline (simulated, lesion
absent) → follow-up (simulated, lesion at ~50% size) → current (the real
exam) — plus printed research features per the plan's requirements:
- **Lesion area change** (absolute and relative)
- **Region overlap (IoU)** between consecutive timepoints
- **Model score change** (if `--checkpoint` given)
- **Registration quality** (number of inlier feature matches — low numbers,
  say under ~15-20, mean registration may be unreliable for that image;
  worth spot-checking those cases)

## What's genuinely real vs. simulated in this pipeline

| Component | Real or simulated? |
|---|---|
| Breast anatomy / tissue texture | REAL (from your actual CBIS-DDSM exam) |
| "Current" timepoint | REAL, completely unmodified |
| Lesion presence/size at baseline & follow-up | SIMULATED |
| Misalignment between visits | SIMULATED (small random rotation/translation) |
| Registration algorithm | REAL — genuinely estimates and corrects alignment |
| Change-detection math (area, IoU, score) | REAL — computed properly on whatever images it's given |

The registration and change-detection code itself is not a toy — I
verified it against a **known synthetic misalignment** (apply a known
rotation+translation, confirm registration recovers it): before
registration, MSE against the true original was 251; after registration,
23 — a 10.5× improvement, using 184 genuine inlier feature matches. That
part of the pipeline works correctly; it's the *input data* (the
timepoints themselves) that's simulated, not the analysis applied to them.

## A bug I caught and fixed while testing — worth knowing about

My first version left a faint but visible ghost of the "removed" lesion in
the baseline image. Cause: my test mask (a hand-drawn rectangle) didn't
exactly match the true circular lesion shape used by the synthetic image
generator, so a sliver of real lesion pixels sat just outside the mask
and never got inpainted away. Fixed by computing the mask with the exact
same geometry formula as the generator. A **very faint** trace can still
remain after inpainting even with a correct mask (OpenCV's inpainting
isn't pixel-perfect on hard-edged synthetic shapes) — visible if you look
closely at the demo figure, honestly disclosed here rather than hidden.
This has no bearing on your real data, where lesion boundaries are
naturally soft rather than a synthetic hard edge.

## Limitations worth stating explicitly in your report

- The simulated lesion trajectory (linear growth, 0% → 50% → 100%) is an
  arbitrary, simplified choice — real lesion growth patterns are not
  necessarily linear or predictable this way.
- Registration is feature-based (ORB + RANSAC affine) and can fail
  silently on low-texture images (check the inlier-match counts printed
  for each patient).
- This module cannot be validated against ground truth, because no ground
  truth (real longitudinal data) exists in this dataset. Its value is
  entirely in demonstrating the pipeline is built correctly, for future
  use with real paired data if it becomes available.