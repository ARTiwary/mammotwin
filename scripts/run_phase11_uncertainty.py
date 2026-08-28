"""
Phase 11 deliverable: uncertainty estimation via MC-Dropout, a data-driven
review threshold, and a check of whether uncertainty actually predicts
model errors.

IMPORTANT: low confidence / high uncertainty means "this case should go to
a human for review" — it must NEVER be silently converted into an
automatic "probably benign" or "probably malignant" verdict. The output of
this script is a three-way split: Prediction, Confidence, and a separate
Needs-Review flag — exactly so those stay visually and logically distinct.

Real data:
    python scripts/run_phase11_uncertainty.py --checkpoint models/baseline_resnet50_<timestamp>.pt

Demo mode (synthetic data + an UNTRAINED model — verifies mechanics only):
    python scripts/run_phase11_uncertainty.py --demo
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

import torch

from src.utils.config import load_config, set_global_seed
from src.data.image_io import load_image, generate_synthetic_mammogram
from src.preprocessing.basic_preprocess import preprocess_image
from src.models.classifier import build_classifier
from src.uncertainty.mc_dropout import mc_dropout_predict
from src.utils.uncertainty_analysis import error_detection_auroc, review_threshold_sweep

LABEL_MAP = {"benign": 0, "malignant": 1}
LABEL_NAMES = {0: "benign", 1: "malignant"}


def load_checkpoint_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_classifier(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def prepare_tensor(path, config, device):
    img = load_image(path)
    result = preprocess_image(img, config, run_quality_gate=False)
    processed = result["processed"]
    tensor = torch.from_numpy(np.ascontiguousarray(processed)).unsqueeze(0)
    tensor = tensor.repeat(3, 1, 1).float().unsqueeze(0).to(device)
    return tensor


def generate_demo_examples(n=30, seed=42):
    import cv2
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        img = generate_synthetic_mammogram(size=300, seed=i)
        path = f"/tmp/mammotwin_phase11_demo_{i}.jpg"
        cv2.imwrite(path, img)
        rows.append({"image_id": f"demo_{i}", "image_file_path_resolved": path,
                      "pathology_binary": rng.choice(["benign", "malignant"])})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 11: uncertainty & review flagging")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--path-col", type=str, default="image_file_path_resolved")
    parser.add_argument("--limit", type=int, default=150,
                         help="Cap on val images (MC-Dropout runs N forward passes PER image)")
    parser.add_argument("--mc-passes", type=int, default=None)
    parser.add_argument("--n-print-examples", type=int, default=10)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.demo:
        print("Running Phase 11 in DEMO mode: synthetic data + an UNTRAINED model.")
        print("Uncertainty numbers here verify the MECHANICS only.\n")
        config = load_config(args.config) if args.config else load_config()
        set_global_seed(config["project"]["seed"])
        model = build_classifier(config).to(device)
        df = generate_demo_examples(n=30, seed=config["project"]["seed"])
    else:
        if not args.checkpoint:
            print("Pass --checkpoint path/to/your/phase6/checkpoint.pt, or use --demo.")
            return
        model, config = load_checkpoint_model(args.checkpoint, device)
        metadata_dir = config["paths"]["data_metadata"]
        val_csv = args.val_csv or os.path.join(metadata_dir, "val_split.csv")
        if not os.path.exists(val_csv):
            print(f"No val CSV at {val_csv}.")
            return
        df = pd.read_csv(val_csv).dropna(subset=[args.path_col, "pathology_binary"])
        df = df.head(args.limit)

    mc_passes = args.mc_passes or config["uncertainty"]["mc_dropout_passes"]
    review_threshold = config["uncertainty"]["review_threshold"]

    print(f"Running MC-Dropout ({mc_passes} passes/image) over {len(df)} images...\n")

    results = []
    for _, row in df.iterrows():
        try:
            tensor = prepare_tensor(row[args.path_col], config, device)
        except Exception:
            continue
        mean_prob, std_prob = mc_dropout_predict(model, tensor, n_passes=mc_passes)
        pred_class = 1 if mean_prob >= 0.5 else 0
        true_class = LABEL_MAP.get(row["pathology_binary"])
        confidence = max(mean_prob, 1 - mean_prob)  # confidence IN the predicted class

        results.append({
            "image_id": row.get("image_id"),
            "true_label": LABEL_NAMES.get(true_class, "?"),
            "predicted_label": LABEL_NAMES[pred_class],
            "mean_prob_malignant": mean_prob,
            "confidence": confidence,
            "mc_dropout_std": std_prob,
            "needs_review": confidence < review_threshold,
            "correct": (pred_class == true_class),
        })

    results_df = pd.DataFrame(results)
    n_total = len(results_df)
    n_review = int(results_df["needs_review"].sum())
    n_errors = int((~results_df["correct"]).sum())

    print(f"Processed {n_total} images.")
    print(f"Flagged for expert review (confidence < {review_threshold}): "
          f"{n_review} ({100 * n_review / n_total:.1f}%)")
    print(f"Model errors in this set: {n_errors} ({100 * n_errors / n_total:.1f}%)\n")

    # --- Does uncertainty actually predict errors? ---
    is_incorrect = (~results_df["correct"]).astype(int).values
    std_auroc = error_detection_auroc(results_df["mc_dropout_std"].values, is_incorrect)
    margin_uncertainty = 1 - results_df["confidence"].values  # simple alternative uncertainty proxy
    margin_auroc = error_detection_auroc(margin_uncertainty, is_incorrect)

    print("=== Does uncertainty predict errors? (AUROC for error detection) ===")
    print("(0.5 = uncertainty is no better than random at flagging errors; higher is better)")
    print(f"  MC-Dropout std as uncertainty score:        "
          f"{std_auroc:.4f}" if std_auroc is not None else "  MC-Dropout std: N/A (no errors or no correct predictions in this set)")
    print(f"  Simple margin (1 - confidence) as score:    "
          f"{margin_auroc:.4f}" if margin_auroc is not None else "  Margin-based: N/A")

    mean_std_correct = results_df.loc[results_df["correct"], "mc_dropout_std"].mean()
    mean_std_incorrect = results_df.loc[~results_df["correct"], "mc_dropout_std"].mean()
    print(f"\n  Mean MC-Dropout std | correct predictions:   {mean_std_correct:.4f}")
    print(f"  Mean MC-Dropout std | incorrect predictions: {mean_std_incorrect:.4f}")
    print("  (Incorrect predictions having a HIGHER mean std is the expected, good pattern.)\n")

    # --- Review threshold sweep (the "set using validation data" step) ---
    print("=== Review threshold sweep (based on THIS validation data) ===")
    sweep_df = review_threshold_sweep(results_df["confidence"].values, is_incorrect)
    print(sweep_df.to_string(index=False))

    metadata_dir = config["paths"]["data_metadata"]
    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    sweep_path = os.path.join(metadata_dir, "phase11_threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    results_path = os.path.join(metadata_dir, "phase11_uncertainty_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {sweep_path}")
    print(f"Saved: {results_path}")

    # --- Figure: confidence distribution, correct vs incorrect ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(results_df.loc[results_df["correct"], "confidence"], bins=15, alpha=0.6,
                 label="Correct", color="#4C72B0")
    axes[0].hist(results_df.loc[~results_df["correct"], "confidence"], bins=15, alpha=0.6,
                 label="Incorrect", color="#C44E52")
    axes[0].axvline(review_threshold, color="black", linestyle="--", label=f"Review threshold ({review_threshold})")
    axes[0].set_xlabel("Confidence in predicted class")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence: correct vs. incorrect predictions")
    axes[0].legend()

    axes[1].plot(sweep_df["threshold"], sweep_df["fraction_of_all_errors_caught"],
                 marker="o", label="Fraction of errors caught")
    axes[1].plot(sweep_df["threshold"], sweep_df["pct_flagged"],
                 marker="s", label="Fraction of cases flagged")
    axes[1].set_xlabel("Review threshold")
    axes[1].set_ylabel("Fraction")
    axes[1].set_title("Review workload vs. errors caught")
    axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "phase11_uncertainty_analysis.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure: {fig_path}")

    # --- The "Prediction / Confidence / Needs Review" demo table ---
    print("\n=== Sample predictions (Prediction / Confidence / Needs Review kept SEPARATE) ===")
    sample = results_df.head(args.n_print_examples)
    for _, r in sample.iterrows():
        review_flag = "-> NEEDS EXPERT REVIEW" if r["needs_review"] else ""
        print(f"  Prediction: {r['predicted_label']:10s} | Confidence: {r['confidence']:.3f} "
              f"| MC-std: {r['mc_dropout_std']:.3f} {review_flag}")

    print("\n" + "=" * 70)
    print("REMINDER: low confidence / high uncertainty means 'send to a human,'")
    print("never 'probably benign.' Flagged cases must not be auto-resolved.")
    print("=" * 70)
    print("\nPhase 11 uncertainty & review flagging COMPLETE.")


if __name__ == "__main__":
    main()