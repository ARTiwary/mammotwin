# Phase 10 — Explainable AI (Grad-CAM)

## Run it

Point at your best Phase 6 whole-image classifier checkpoint:

```powershell
python scripts\run_phase10_gradcam.py --checkpoint models\baseline_resnet50_20260825_114630.pt
```

(swap in whatever your actual best checkpoint filename is — check
`models\registry.json` if unsure which one had the highest `val_auc`)

Add the optional experimental counterfactual check (masks the most-attended
region and re-runs the model, to see how much the prediction shifts):

```powershell
python scripts\run_phase10_gradcam.py --checkpoint models\baseline_resnet50_20260825_114630.pt --run-counterfactual
```

## What it produces

`reports\figures\phase10_gradcam_examples.png` — a grid of examples (mixed
correct and incorrect predictions where available), each showing:
- The preprocessed input, with the **ground-truth lesion box in green**
  (pulled from your real Phase 7 bbox data, transformed into the same
  coordinate frame the model actually saw — reuses the exact crop+resize
  transform validated in Phase 7)
- The Grad-CAM heatmap overlay, same box drawn on top, so you can see
  directly whether the model's attention lines up with the actual lesion

This is the single most useful figure this phase produces for your
report/paper: cases where the heatmap overlaps the green box are evidence
the model is (at least partly) using real lesion signal; cases where it's
firing somewhere else entirely — especially on WRONG predictions — are
worth discussing as a limitation.

## Why Grad-CAM was implemented from scratch, not via a library

After hitting two real dependency issues in this project (`torchmetrics`
pulling in a `scipy` DLL blocked by your Windows security policy in Phase
7, and choosing to avoid MONAI for the same reason in Phase 8), I wrote
Grad-CAM directly rather than adding the `grad-cam` package as a third
dependency. It's ~100 lines, and now you can describe the exact algorithm
in your methods section instead of citing an opaque library.

## What I verified vs. what needs your real checkpoint

**Verified:** the full mechanics run end-to-end — forward/backward hooks,
heatmap generation, normalization, colormap overlay, ground-truth box
coordinate transform, and the counterfactual masking experiment. I tested
this with a demo mode using an **untrained, randomly-initialized model** —
sufficient to prove the code is correct, but the resulting heatmaps carry
no real diagnostic meaning (a random model has no learned class
discrimination to visualize).

**Needs your real data:** whether your actual trained model's attention
meaningfully overlaps real lesions. That's the entire point of this phase,
and it can only be checked with your real checkpoint.

## The disclaimer, stated plainly (also printed by the script itself)

A Grad-CAM heatmap shows where the model's evidence was concentrated. It
is a model explanation, not proof of cancer, and does not by itself
validate that a prediction is correct — a wrong prediction can still have
a heatmap that "looks reasonable," and a correct prediction can be right
for spurious reasons the heatmap won't necessarily reveal clearly. Treat
this as a diagnostic and communication tool, not a certification.

## Notes on the counterfactual experiment

This is explicitly exploratory (per the project plan's own wording) — it
tells you "does masking the model's top-attended region reduce its
confidence in this class," which is suggestive but not a rigorous causal
attribution method. A large probability drop after masking is a mildly
reassuring sign the model relies on that region; a small or negative drop
doesn't necessarily mean the region was irrelevant (the model may have
redundant evidence elsewhere in the image).