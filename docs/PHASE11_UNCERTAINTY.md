# Phase 11 — Uncertainty & Human Review

## Run it

```powershell
python scripts\run_phase11_uncertainty.py --checkpoint models\baseline_resnet50_20260825_114630.pt
```

MC-Dropout runs `mc_dropout_passes` (default 20, from `config.yaml`)
forward passes **per image** — this is meaningfully slower than a normal
inference pass. `--limit 150` (the default) caps it to the first 150
validation images to keep the first run fast; raise it once you've
confirmed the output looks reasonable:

```powershell
python scripts\run_phase11_uncertainty.py --checkpoint models\baseline_resnet50_20260825_114630.pt --limit 500 --mc-passes 30
```

## What it produces

- **Printed AUROC for error detection** — the key scientific question of
  this phase: does higher uncertainty actually predict wrong predictions?
  0.5 = uncertainty is useless (no better than random); meaningfully above
  0.5 = uncertainty is doing real work. Two versions are computed and
  compared: MC-Dropout standard deviation, and a simpler "distance from
  0.5" margin — worth reporting both, since it's an interesting finding
  either way if one clearly outperforms the other.
- **A review-threshold sweep table** (`data\metadata\phase11_threshold_sweep.csv`)
  — at each candidate confidence threshold: what fraction of cases get
  flagged, the error rate inside vs. outside the flagged group, and what
  fraction of ALL errors get caught. This is the concrete, data-driven
  answer to "how do I pick a review threshold" — pick the point on this
  table that trades off review workload against errors caught in a way
  that makes sense for your write-up.
- `reports\figures\phase11_uncertainty_analysis.png` — confidence
  distributions for correct vs. incorrect predictions (should be visibly
  different if uncertainty is meaningful), and the threshold-sweep chart.
- `data\metadata\phase11_uncertainty_results.csv` — full per-image results.
- A printed sample table keeping **Prediction, Confidence, and Needs-Review
  strictly separate columns**, per the plan's explicit framing — this
  script never collapses low confidence into an automatic verdict.

## Why MC-Dropout specifically

The plan calls this out as an "experimental uncertainty method," which is
the right framing — MC-Dropout is cheap (no retraining, no ensemble of
separate models needed) but a known-imperfect approximation to true
Bayesian uncertainty. It reuses the Dropout layer already present in your
classifier's head (`build_classifier`'s final `Dropout(dropout)` layer from
Phase 6) — no architecture changes needed.

## What I verified vs. what needs your real checkpoint

**Verified:** MC-Dropout produces genuine, non-trivial variation across
passes (std in a sensible 0.05–0.08 range on the untrained demo model, not
zero and not absurd); the review-flagging logic keeps Prediction/
Confidence/Needs-Review properly separate; the threshold sweep handles the
edge case where every example lands on one side (no unflagged group)
without crashing; AUROC computation and the figure both render correctly.

**Needs your real checkpoint:** whether MC-Dropout uncertainty on your
actual trained model meaningfully separates correct from incorrect
predictions — that's the entire point, and only real data can answer it.
The demo model is untrained/random, so everything clusters near 0.5
confidence and gets flagged — expected, not a bug.

## The rule this phase exists to enforce

Never convert "the model isn't confident" into "probably benign." A
flagged case is a signal to route to a human, full stop — not a soft
verdict of its own. The script's printed reminder and its column layout
(Prediction / Confidence / Needs Review always shown as separate fields,
never merged into one number) are both designed to make that mistake
structurally awkward to make by accident.