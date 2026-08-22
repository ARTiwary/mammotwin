# MammoTwin — Project Charter (Phase 1: Requirements & Planning)

## 1. Problem Statement

Standard mammogram AI systems collapse a rich diagnostic workflow into a single
binary output (cancer / no cancer), giving no indication of *where* the model
looked, *how confident* it is, or *how the case has changed* over time. This
opacity limits clinical trust and research usefulness.

MammoTwin is an academic research prototype that reframes mammogram analysis
as a multi-stage pipeline — quality check → localization → segmentation →
classification → uncertainty estimation → explainability → optional
prior-vs-current comparison — culminating in a single research dashboard.

**This is not a clinical diagnostic system.** No output may be presented as a
medical diagnosis. All outputs are for research/educational demonstration only.

## 2. Objectives

1. Build a reproducible, patient-level-split pipeline for mammogram lesion
   classification using CBIS-DDSM.
2. Localize and (where valid masks exist) segment suspicious regions rather
   than relying on whole-image classification alone.
3. Make every prediction explainable (Grad-CAM) and uncertainty-aware
   (flag low-confidence cases for expert review instead of forcing an answer).
4. Prototype a longitudinal (prior-vs-current) comparison module, using real
   paired data if available, otherwise clearly-labeled simulated data.
5. Package the pipeline behind a FastAPI backend and React dashboard as a
   convincing, end-to-end research demo.
6. Produce a paper-ready report with rigorous evaluation, ablations, and an
   explicit discussion of limitations (patient-level leakage prevention,
   dataset bias, calibration).

## 3. Prediction Target

Defined from CBIS-DDSM documentation: pathology-based classification per
finding (benign / malignant), at the finding/image level, evaluated with
patient-level splitting to prevent leakage. (Confirm exact label field —
`pathology` — and calcification vs. mass subsets during Phase 3.)

## 4. Scope

### Minimum Viable Product (MVP)
- Baseline classifier (whole-image, transfer learning)
- Grad-CAM explainability
- Calibrated confidence + low-confidence review flag
- Basic dashboard: upload → prediction → heatmap → confidence

### Advanced Modules (post-MVP)
- Lesion localization (bounding boxes)
- Lesion segmentation (where valid masks exist)
- Lesion-crop classification upgrade
- Longitudinal prior-vs-current comparison (real or clearly-labeled simulated)
- Optional multimodal fusion (image + structured variables)

## 5. Evaluation Metrics (fixed before training)

| Module | Metrics |
|---|---|
| Classification | ROC-AUC, PR-AUC, sensitivity, specificity, precision, F1, balanced accuracy |
| Calibration | Reliability diagram, Brier score |
| Localization | IoU, mAP, sensitivity/false-positive rate |
| Segmentation | Dice, IoU |
| Explainability | Qualitative overlay review, annotation-overlap where valid |
| Uncertainty | Error rate vs. confidence correlation, flagged-case precision |
| Robustness | Performance on external dataset (e.g. INbreast), if label-compatible |

Golden rule: tune only on validation data; touch the locked test set exactly
once, at the end (Phase 14).

## 6. System Architecture

```
User
  │
  ▼
Mammogram Upload
  │
  ▼
Image / DICOM Handler
  │
  ▼
Quality Gate ──(reject)──► "Unsuitable for analysis"
  │
  ▼
Preprocessing (normalize, crop, resize)
  │
  ├──► Localization ──► Segmentation (if valid masks)
  │
  ▼
Classification
  │
  ▼
Calibration / Uncertainty ──► "Needs expert review" flag
  │
  ▼
Explainability (Grad-CAM)
  │
  ▼
Optional Longitudinal Comparison (prior vs. current)
  │
  ▼
Dashboard / Research Report
```

Backend: FastAPI (stateless `/predict`, `/explain`, `/uncertainty`,
`/longitudinal` endpoints, Pydantic response schemas).
Frontend: React (upload panel + result panels per pipeline stage).

## 7. Dataset & Safety / Data-Use Statement

- **Primary dataset:** CBIS-DDSM — curated mammography cases with lesion
  information and pathology labels. License/terms to be reviewed and recorded
  in `data/metadata/DATASET_LICENSE_NOTES.md` before download.
- **Supplementary:** INbreast (external/robustness testing), VinDr-Mammo
  (detection/BI-RADS), MIAS (small intro experiments only).
- **Longitudinal data:** CBIS-DDSM is *not* a longitudinal dataset. The
  MammoTwin temporal module will only use genuine paired prior/current exams
  if a legitimately licensed longitudinal dataset is obtained; otherwise it
  will run on explicitly labeled **simulated** pairs for demonstration only.
- **Splitting rule:** All splits are performed at patient/study level, never
  at image level, to prevent leakage. Test set is locked after Phase 3 and
  used exactly once (Phase 14).
- **Redistribution:** No patient images will be published or redistributed
  beyond what dataset terms explicitly allow. Only code and permitted
  metadata are shared publicly.
- **Clinical disclaimer:** Every dashboard output will carry a visible
  statement that this is a research prototype, not a diagnostic tool, and
  that outputs must not be used for real patient care decisions.

## 8. Timeline (12 Weeks)

| Week | Focus |
|---|---|
| 1 | Requirements, literature review, dataset access, environment setup |
| 2 | Dataset exploration, patient-level splitting |
| 3 | Preprocessing and quality checks |
| 4 | Baseline classifier |
| 5 | Model tuning and evaluation |
| 6 | Localization |
| 7 | Segmentation or lesion-crop classification |
| 8 | Explainability (Grad-CAM) |
| 9 | Uncertainty / calibration |
| 10 | Longitudinal comparison prototype |
| 11 | Dashboard integration (FastAPI + React) and testing |
| 12 | Final experiments, documentation, presentation, demo |

## 9. Repository & Tooling

- Git repository initialized (`mammotwin/`), `.gitignore` excludes raw data,
  model weights, and node/venv artifacts.
- Config-driven paths/seed/hyperparameters (`config/config.yaml`) — single
  source of truth so training and inference never drift apart.
- Experiment log (`reports/eval_results/experiments_log.csv`) started from
  Phase 6 onward: every training run's config, seed, and metrics recorded for
  the ablation/results section of the paper.
