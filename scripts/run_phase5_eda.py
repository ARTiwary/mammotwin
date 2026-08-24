"""
Phase 5 deliverable: Exploratory Data Analysis.

Produces:
  - reports/figures/phase5_class_distribution.png
  - reports/figures/phase5_image_dimensions.png
  - reports/figures/phase5_representative_cases.png
  - reports/figures/phase5_cropped_examples.png
  - data/metadata/phase5_crosstabs.csv  (view/laterality/density vs. pathology)
  - reports/EDA_SUMMARY.md  (written observations, auto-filled with real numbers)

Usage (real data):
    python scripts/run_phase5_eda.py

Demo mode (synthetic data, no real CBIS-DDSM needed):
    python scripts/run_phase5_eda.py --demo
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import load_config, set_global_seed
from src.data.image_io import load_image, generate_synthetic_mammogram


def plot_class_distribution(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    counts = df["pathology_binary"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["#4C72B0", "#C44E52"])
    axes[0].set_title("Pathology distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts.values) * 0.01, str(v), ha="center")

    if "finding_type" in df.columns:
        cross = pd.crosstab(df["finding_type"], df["pathology_binary"])
        cross.plot(kind="bar", stacked=True, ax=axes[1],
                   color=["#4C72B0", "#C44E52"])
        axes[1].set_title("Pathology by finding type")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return counts


def plot_image_dimensions(df: pd.DataFrame, path_col: str, out_path: str, sample_size: int = 100, seed: int = 42):
    sample = df.dropna(subset=[path_col]).sample(
        n=min(sample_size, df[path_col].notna().sum()), random_state=seed
    )
    widths, heights = [], []
    n_load_failures = 0
    for path in sample[path_col]:
        try:
            img = load_image(path)
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
        except Exception:
            n_load_failures += 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if widths:
        axes[0].scatter(widths, heights, alpha=0.5, s=15)
        axes[0].set_xlabel("Width (px)")
        axes[0].set_ylabel("Height (px)")
        axes[0].set_title(f"Image dimensions (n={len(widths)} sampled)")

        axes[1].hist(np.array(widths) / np.array(heights), bins=20, color="#4C72B0")
        axes[1].set_xlabel("Aspect ratio (width / height)")
        axes[1].set_title("Aspect ratio distribution")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No images could be loaded", ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "n_sampled": len(sample),
        "n_load_failures": n_load_failures,
        "width_range": (min(widths), max(widths)) if widths else None,
        "height_range": (min(heights), max(heights)) if heights else None,
    }


def plot_representative_cases(df: pd.DataFrame, path_col: str, out_path: str, seed: int = 42):
    rng = np.random.default_rng(seed)
    groups = []
    for finding_type in df["finding_type"].dropna().unique() if "finding_type" in df.columns else [None]:
        for pathology in ["benign", "malignant"]:
            subset = df[df["pathology_binary"] == pathology]
            if finding_type is not None:
                subset = subset[subset["finding_type"] == finding_type]
            subset = subset.dropna(subset=[path_col])
            if len(subset) > 0:
                row = subset.sample(n=1, random_state=rng.integers(0, 1_000_000)).iloc[0]
                label = f"{finding_type or ''} {pathology}".strip()
                groups.append((label, row[path_col]))

    if not groups:
        return None

    n = len(groups)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (label, path) in enumerate(groups):
        try:
            img = load_image(path)
            axes[i].imshow(img, cmap="gray")
        except Exception as e:
            axes[i].text(0.5, 0.5, f"load failed:\n{e}", ha="center", va="center", fontsize=8)
        axes[i].set_title(label)
        axes[i].axis("off")

    for j in range(len(groups), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return groups


def plot_cropped_examples(df: pd.DataFrame, out_path: str, n_examples: int = 3, seed: int = 42):
    """Show full image next to its cropped lesion patch, where available."""
    col = "cropped_image_file_path_resolved"
    if col not in df.columns:
        return None

    subset = df.dropna(subset=["image_file_path_resolved", col]).copy()
    if "cropped_image_file_path_n_candidates" in df.columns:
        # Prefer unambiguous (single-candidate) cropped paths for a cleaner example.
        unambiguous = subset[subset["cropped_image_file_path_n_candidates"] <= 1]
        subset = unambiguous if len(unambiguous) > 0 else subset

    if len(subset) == 0:
        return None

    sample = subset.sample(n=min(n_examples, len(subset)), random_state=seed)
    fig, axes = plt.subplots(2, len(sample), figsize=(4 * len(sample), 8))
    if len(sample) == 1:
        axes = axes.reshape(2, 1)

    for i, (_, row) in enumerate(sample.iterrows()):
        try:
            full_img = load_image(row["image_file_path_resolved"])
            axes[0, i].imshow(full_img, cmap="gray")
        except Exception:
            axes[0, i].text(0.5, 0.5, "load failed", ha="center", va="center")
        axes[0, i].set_title(f"Full: {row.get('pathology_binary', '')}")
        axes[0, i].axis("off")

        try:
            crop_img = load_image(row[col])
            axes[1, i].imshow(crop_img, cmap="gray")
        except Exception:
            axes[1, i].text(0.5, 0.5, "load failed", ha="center", va="center")
        axes[1, i].set_title("Cropped lesion patch")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return len(sample)


def compute_shortcut_crosstabs(df: pd.DataFrame) -> dict:
    """Cross-tabulate metadata fields against pathology to check whether any
    non-clinical field (view, laterality, density) looks suspiciously
    predictive on its own — a sign of a potential shortcut/artifact rather
    than genuine lesion signal."""
    results = {}
    for col in ["view", "laterality", "breast_density"]:
        if col in df.columns:
            cross = pd.crosstab(df[col], df["pathology_binary"], normalize="index")
            results[col] = cross
    return results


def generate_synthetic_eda_dataset(n: int = 40, seed: int = 42):
    """Small synthetic metadata + matching images, purely to exercise every
    EDA code path before running for real."""
    rng = np.random.default_rng(seed)
    rows = []
    tmp_dir = "/tmp/mammotwin_phase5_demo_images"
    os.makedirs(tmp_dir, exist_ok=True)

    import cv2
    for i in range(n):
        pathology = rng.choice(["benign", "malignant"], p=[0.6, 0.4])
        finding_type = rng.choice(["mass", "calc"])
        view = rng.choice(["CC", "MLO"])
        laterality = rng.choice(["LEFT", "RIGHT"])
        density = rng.integers(1, 5)

        img_path = os.path.join(tmp_dir, f"img_{i}.jpg")
        img = generate_synthetic_mammogram(size=rng.choice([400, 512, 600]), seed=i)
        cv2.imwrite(img_path, img)

        rows.append({
            "patient_id": f"P_{i:04d}",
            "finding_type": finding_type,
            "pathology_binary": pathology,
            "view": view,
            "laterality": laterality,
            "breast_density": density,
            "image_file_path_resolved": img_path,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 5: EDA")
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--path-col", type=str, default="image_file_path_resolved")
    parser.add_argument("--sample-size", type=int, default=100,
                         help="How many images to sample for dimension analysis")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["project"]["seed"])
    seed = config["project"]["seed"]
    metadata_dir = config["paths"]["data_metadata"]
    figures_dir = config["paths"]["figures_dir"]
    reports_dir = config["paths"]["reports_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    if args.demo:
        print("Running Phase 5 EDA in DEMO mode on synthetic data.\n")
        df = generate_synthetic_eda_dataset(n=40, seed=seed)
    else:
        metadata_csv = args.metadata_csv or os.path.join(metadata_dir, "metadata.csv")
        if not os.path.exists(metadata_csv):
            print(f"No metadata CSV found at {metadata_csv}. Run Phase 3 first, "
                  f"or pass --demo.")
            return
        df = pd.read_csv(metadata_csv)

    print(f"Loaded {len(df)} rows, {df['patient_id'].nunique()} patients.\n")

    print("=== Class distribution ===")
    counts = plot_class_distribution(df, os.path.join(figures_dir, "phase5_class_distribution.png"))
    print(counts)
    imbalance_ratio = counts.max() / counts.min()
    print(f"Imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1\n")

    print("=== Image dimensions (sampled) ===")
    dim_stats = plot_image_dimensions(df, args.path_col,
                                       os.path.join(figures_dir, "phase5_image_dimensions.png"),
                                       sample_size=args.sample_size, seed=seed)
    print(dim_stats, "\n")

    print("=== Representative cases ===")
    groups = plot_representative_cases(df, args.path_col,
                                        os.path.join(figures_dir, "phase5_representative_cases.png"), seed=seed)
    print(f"Saved {len(groups) if groups else 0} representative examples.\n")

    print("=== Cropped lesion examples ===")
    n_cropped = plot_cropped_examples(df, os.path.join(figures_dir, "phase5_cropped_examples.png"), seed=seed)
    if n_cropped:
        print(f"Saved {n_cropped} full/cropped example pairs.\n")
    else:
        print("Skipped (no cropped_image_file_path_resolved column, or none available).\n")

    print("=== Shortcut / artifact audit ===")
    crosstabs = compute_shortcut_crosstabs(df)
    crosstab_rows = []
    for field, cross in crosstabs.items():
        print(f"\n{field} vs. pathology (row-normalized):")
        print(cross)
        cross_reset = cross.reset_index()
        cross_reset.insert(0, "field", field)
        crosstab_rows.append(cross_reset)
    if crosstab_rows:
        combined = pd.concat(crosstab_rows, ignore_index=True)
        crosstab_path = os.path.join(metadata_dir, "phase5_crosstabs.csv")
        combined.to_csv(crosstab_path, index=False)
        print(f"\nSaved crosstabs: {crosstab_path}")

    # --- Write summary markdown ---
    summary_path = os.path.join(reports_dir, "EDA_SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write("# MammoTwin — Phase 5 EDA Summary\n\n")
        f.write(f"Dataset: {len(df)} rows, {df['patient_id'].nunique()} patients.\n\n")
        f.write("## Class distribution\n\n")
        f.write(counts.to_frame("count").to_markdown() + "\n\n")
        f.write(f"Imbalance ratio (majority:minority): **{imbalance_ratio:.2f}:1** — ")
        if imbalance_ratio < 2:
            f.write("mild, standard class weighting should suffice.\n\n")
        else:
            f.write("notable — consider class-weighted loss or balanced sampling in Phase 6.\n\n")
        f.write("## Image dimensions\n\n")
        f.write(f"Sampled {dim_stats['n_sampled']} images "
                f"({dim_stats['n_load_failures']} failed to load). ")
        if dim_stats["width_range"]:
            f.write(f"Width range: {dim_stats['width_range']}, "
                    f"height range: {dim_stats['height_range']}.\n\n")
        f.write("## Shortcut / artifact audit\n\n")
        f.write("Cross-tabulated view, laterality, and breast density against pathology "
                "(see `data/metadata/phase5_crosstabs.csv`). Any field where one category "
                "shows a malignancy rate far outside the overall base rate is worth a second "
                "look before training — it may indicate an acquisition artifact the model "
                "could learn as a shortcut rather than genuine lesion signal.\n\n")
        f.write(f"Overall malignant rate: {counts.get('malignant', 0) / counts.sum():.1%}\n\n")
        f.write("## Figures\n\n")
        f.write("- `phase5_class_distribution.png`\n"
                "- `phase5_image_dimensions.png`\n"
                "- `phase5_representative_cases.png`\n"
                "- `phase5_cropped_examples.png`\n")

    print(f"\nSaved EDA summary: {summary_path}")
    print("\nPhase 5 EDA complete.")


if __name__ == "__main__":
    main()