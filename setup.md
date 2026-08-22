# Phase 2 — Environment Setup

Run these from the repo root (the `mammotwin/` folder on your machine).

## 1. Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA-accelerated PyTorch instead of the
CPU build, install torch/torchvision separately **before** the rest of
`requirements.txt`, using the exact command from
https://pytorch.org/get-started/locally/ for your CUDA version, e.g.:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt   # installs everything else
```

## 3. Verify the environment

Run the Phase 2 deliverable script — it checks that every core library
imports correctly, then loads, preprocesses, and displays one image
end-to-end:

```bash
python scripts/verify_environment.py
```

This runs on a synthetic placeholder image by default (no real data needed
yet — that's Phase 3). Once you have a real mammogram file to test with:

```bash
python scripts/verify_environment.py --image path/to/some_image.png
python scripts/verify_environment.py --image path/to/some_image.dcm
```

Expected output: version numbers for numpy/opencv/matplotlib/pydicom/torch,
raw vs. processed image shapes, and a saved figure at
`reports/figures/phase2_preprocessing_check.png`.

## 4. Git

If you haven't already:
```bash
git init
git add -A
git commit -m "Phase 2: environment setup, config, preprocessing pipeline verified"
```

## Phase 2 checklist

- [ ] Virtual environment created and activated
- [ ] `pip install -r requirements.txt` completes with no errors
- [ ] `python scripts/verify_environment.py` prints "Phase 2 environment check PASSED"
- [ ] `reports/figures/phase2_preprocessing_check.png` generated and looks correct
- [ ] Changes committed to Git

Once this checklist is done, move on to **Phase 3 — Data Acquisition &
Understanding** (downloading CBIS-DDSM and building the metadata table).