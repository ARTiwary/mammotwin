# Phase 6 — Baseline Classifier

## 1. Install PyTorch

If you haven't already (Phase 2's `requirements.txt` includes CPU torch by
default). If you have an NVIDIA GPU, install the CUDA build first — it will
make training dramatically faster:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Check it's using your GPU:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Smoke test on real data first (recommended)

Before committing to a full training run, do a quick sanity check on a
small slice of your real data — this catches path/loading issues in
minutes instead of after a long run:

```powershell
python scripts\run_phase6_baseline.py --epochs 2 --limit 100
```

You should see class counts printed, decreasing train loss, and a
`val_auc` printed each epoch (won't be meaningful with only 2 epochs/100
images — this step is just to confirm nothing crashes).

## 3. Full baseline training run

```powershell
python scripts\run_phase6_baseline.py
```

Uses everything from `config\config.yaml` (30 epochs, batch size 16,
resnet50, early stopping patience 5) unless overridden:

```powershell
python scripts\run_phase6_baseline.py --epochs 30 --batch-size 32 --lr 0.0001
```

On CPU, this will be slow — a full pass over ~2500 training images with
ResNet50 could take a long while per epoch. If you don't have a GPU,
consider:
- Switching to `resnet18` in `config\config.yaml` (`model: backbone:`) —
  much faster, a reasonable first baseline.
- Reducing `image_size` in `config.yaml` (e.g. `[160, 160]`) for a faster
  first pass, then increasing later for the "real" run.

## 4. What gets produced

- `models\<run_id>.pt` — the best checkpoint (by validation ROC-AUC)
- `models\registry.json` — every trained model's version, backbone, val
  AUC, and training date
- `reports\eval_results\experiments_log.csv` — one row appended per run
  (config used, val metrics) — this is what feeds your paper's
  experiments/ablation table later
- `reports\figures\phase6_training_curves.png` — loss + val AUC curves
- Printed classification report: accuracy, balanced accuracy, sensitivity,
  specificity, precision, F1, ROC-AUC, PR-AUC, confusion matrix — all
  computed on the **validation** set only (test set stays locked until
  Phase 14)

## 5. What "done" looks like for this phase

Per the plan: don't move to Phase 7 (localization) until this baseline
trains cleanly and produces a genuine, reasonable evaluation report on
real data — not just a script that runs without crashing.

A few honest benchmarks to sanity-check against: published CBIS-DDSM
whole-image classification baselines typically land somewhere in the
0.65–0.80 ROC-AUC range depending on architecture/preprocessing choices.
If your first real run is close to 0.5 (random), something's likely wrong
(check: are class labels mapped correctly, is normalization sane, is the
model actually training — is train_loss decreasing at all). If it's
suspiciously high (>0.95) on the very first baseline, be suspicious of
leakage before celebrating.

## 6. Known limitations at this stage (expected, not bugs)

- Quality-flagged images from Phase 4's `quality_report.csv` are NOT yet
  automatically excluded from training — that wiring is worth adding
  before your final run (filter `train_split.csv` down to `image_id`s
  where `passed == True`, or extend `MammogramDataset` to accept an
  exclusion list).
- No data augmentation yet (flips, rotations, brightness jitter) —
  reasonable for a first baseline, but likely worth adding via
  Albumentations for the Phase 9 classification upgrade.
- Batch size 16 / image size 224×224 is a starting point, not a tuned
  choice — Phase 6's "model tuning" sub-step is exactly where you'd sweep
  these once the first run completes successfully.