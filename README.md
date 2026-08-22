# MammoTwin

An explainable, uncertainty-aware research prototype for mammogram lesion
analysis and longitudinal change monitoring.

> **Safety notice:** This is an academic/research prototype, not a clinical
> diagnostic system. Outputs must never be presented as a medical diagnosis.

## Status

Phase 1 (Requirements & Planning) — see [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md)
for the full problem statement, scope, architecture, evaluation plan, timeline,
and data-use/safety statement.

## Repository Layout

```
config/         Central config (paths, seed, hyperparameters)
data/           raw / processed / metadata / external — see .gitignore, no patient data committed
notebooks/      EDA and analysis notebooks
src/            Core ML pipeline (data, preprocessing, models, explainability, uncertainty, longitudinal)
models/         Saved checkpoints + registry.json (not committed)
backend/        FastAPI service
frontend/       React dashboard
reports/        Figures, evaluation results, paper draft
tests/          Unit tests
docs/           Planning documents (this phase's deliverables)
```

## Tech Stack

Python, PyTorch, OpenCV, pydicom, MONAI, NumPy, pandas, scikit-learn,
Matplotlib, Albumentations, FastAPI, React.

## Dataset

Primary: **CBIS-DDSM**. Supplementary: INbreast, VinDr-Mammo, MIAS.
See the charter for licensing and longitudinal-data caveats.

## Development Order

Follow the 17-phase plan in `docs/PROJECT_CHARTER.md`. Do not start the
FastAPI/React app until the baseline classifier (Phase 6) is trained and
evaluated on a locked, patient-level test split.
