# Phase 13 — Optional Multimodal Model

## The most important design decision in this phase

CBIS-DDSM's `assessment` column (the radiologist's own BI-RADS suspicion
category, 0-6) is **deliberately excluded** from the structured features.
It's not an independent variable — it's the clinician's own near-diagnosis,
made by looking at the same image and often directly informing (or
directly reflecting) the pathology outcome. Including it as a predictor
would be textbook label leakage: the model would partly just be learning
to decode the radiologist's own conclusion rather than finding independent
signal. The plan is explicit about avoiding exactly this
("avoid variables that directly leak the target label"), and this is the
clearest candidate for it in this dataset.

**What's used instead:** `breast_density`, `laterality`, `view`,
`finding_type` (mass/calc), `subtlety`, and the morphological descriptors
(`mass_shape`, `mass_margins`, `calc_type`, `calc_distribution`). Worth
noting honestly in your report: some of these descriptors (e.g. "irregular"
shape, "spiculated" margins) are themselves part of the BI-RADS suspicion
lexicon and carry real signal for that reason — they describe what a
lesion looks like, not a diagnostic verdict, which is a meaningfully
different (and more defensible) category than `assessment` itself, but
it's not a perfectly clean line and worth acknowledging as a nuance.

## Run it

```powershell
python scripts\run_phase13_multimodal.py --epochs 30
```

## What it produces

- A trained fusion model: CNN image embedding + small MLP over the
  structured features, concatenated and passed through a final
  classification head.
- The same evaluation suite as Phase 6/9: full classification metrics,
  calibration (reliability diagram + Brier score), checkpoint, registry
  entry, and experiment log row (`phase: 13_multimodal`).
- **An automatic comparison against Phase 6's whole-image-only baseline** —
  the actual research question this phase answers: does adding structured
  metadata improve on image alone?

## Careful missing-value handling (per the plan's explicit requirement)

- **Categorical columns**: missing values get an explicit `"missing"`
  category rather than being imputed to some arbitrary mode. For CBIS-DDSM
  specifically, `mass_shape`/`mass_margins` are missing for calcification
  findings and `calc_type`/`calc_distribution` are missing for mass
  findings **by definition** — a finding is one or the other, never both —
  so "missing" here is actually informative (it tells the model the
  finding type), not a data-quality gap to paper over.
- **Numeric columns** (`breast_density`, `subtlety`): median-imputed using
  only the **training set's** median (never val/test — that would leak
  information), plus an explicit `was_missing` indicator column, then
  standardized using the training set's mean/std.
- I tested this directly: unseen categories at inference time (never seen
  during training) fall back safely to `"missing"` rather than crashing or
  silently misassigning — confirmed with a dedicated test before building
  anything on top of it.

## What I verified vs. what needs your real data

**Verified:** the tabular preprocessing (missing values, unseen categories,
train/val dimensional consistency), the forward pass shapes through the
fusion architecture, and the full training/evaluation/comparison pipeline
end-to-end in demo mode.

**Needs your real data:** whether structured metadata actually improves
over whole-image-only on your real CBIS-DDSM data. Given Phase 9 and
Phase 12 both pointed toward this classifier relying heavily on
whole-image context already, my honest prior is that the marginal
improvement from adding a handful of categorical/numeric fields may be
modest — but that's exactly the kind of prediction real data should be
allowed to overturn, not something to assume going in.