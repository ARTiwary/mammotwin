# Phase 3 — Data Acquisition & Understanding

## 1. Getting CBIS-DDSM

Two common routes — check current license/terms on whichever you use, and
record them in `data/metadata/DATASET_LICENSE_NOTES.md`:

**Option A — Official (TCIA):**
1. Go to the CBIS-DDSM collection page on The Cancer Imaging Archive (TCIA).
2. Download the manifest file and use the NBIA Data Retriever tool to pull
   the DICOM images.
3. Download the four case-description CSVs (mass/calc × train/test) from
   the same collection page.

**Option B — Kaggle mirror:**
1. Search Kaggle for "CBIS-DDSM breast cancer image dataset."
2. Confirm the mirror's listed license before use — Kaggle mirrors sometimes
   repackage images (JPEG re-encoded) rather than original DICOM; note this
   in `DATASET_LICENSE_NOTES.md` since it affects Phase 4 preprocessing
   (no DICOM inversion handling needed if images are already JPEG/PNG).

Either way, **do not commit or redistribute the raw images** — `data/raw/`
is already excluded via `.gitignore`.

## 2. Build the unified metadata table

Once you have the four CSVs:

```bash
python -m src.data.build_metadata \
  --mass-train path/to/mass_case_description_train_set.csv \
  --mass-test  path/to/mass_case_description_test_set.csv \
  --calc-train path/to/calc_case_description_train_set.csv \
  --calc-test  path/to/calc_case_description_test_set.csv \
  --output data/metadata/metadata.csv
```

This normalizes both mass and calc CSVs onto one schema (they use slightly
different column names — e.g. `mass shape` vs `calc type`) and derives:
- `pathology_binary`: benign / malignant (BENIGN_WITHOUT_CALLBACK → benign)
- `image_id`: a stable per-row identifier

## 3. Run checks + produce the locked patient-level split

```bash
python scripts/run_phase3.py \
  --mass-train path/to/mass_case_description_train_set.csv \
  --mass-test  path/to/mass_case_description_test_set.csv \
  --calc-train path/to/calc_case_description_train_set.csv \
  --calc-test  path/to/calc_case_description_test_set.csv \
  --raw-images-dir path/to/CBIS-DDSM/images
```

This will:
1. Build `data/metadata/metadata.csv`
2. Run integrity checks: class balance, duplicate IDs, missing files
3. Produce a **patient-level** train/val/test split (never image-level)
4. **Assert** no patient appears in more than one split — this will hard-fail
   loudly if violated, by design
5. Save `data/metadata/{train,val,test}_split.csv`

Don't have real data yet? Run in demo mode to see the exact same pipeline
work on synthetic data:

```bash
python scripts/run_phase3.py --demo
```

## 4. Manual review checklist

After running the script, still eyeball:

- [ ] Does `pathology_binary` class balance roughly match published
      CBIS-DDSM statistics (dataset is moderately imbalanced toward benign)?
- [ ] Do view (`CC`/`MLO`) and laterality (`LEFT`/`RIGHT`) counts look sane?
- [ ] Spot-check 5–10 `image_file_path` entries actually open (Phase 4 will
      formalize this, but catch obvious path issues now).
- [ ] Confirm `test_split.csv` patient count is reasonable (~15% of patients)
      and note it as **locked** — no further tuning against it until Phase 14.

## 5. Once this phase is done

Commit `data/metadata/metadata.csv` and the three split CSVs (these are
small, don't contain images, and are exactly what dataset terms typically
permit to share) — but double check your specific CBIS-DDSM license terms
first.