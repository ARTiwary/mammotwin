"""
Phase 13 deliverable: train the multimodal (image + structured data fusion)
classifier and compare it against Phase 6's whole-image-only baseline.

IMPORTANT: the CBIS-DDSM 'assessment' column (the radiologist's own BI-RADS
suspicion category) is DELIBERATELY EXCLUDED from the structured features.
It is the clinician's own near-diagnosis, not an independent variable —
including it would be textbook label leakage (predicting pathology using a
feature that already encodes the radiologist's conclusion about pathology),
exactly what the plan warns against ("avoid variables that directly leak
the target label").

Structured features used instead: breast_density, laterality, view,
finding_type (mass/calc), subtlety, and the morphological descriptors
(mass_shape, mass_margins, calc_type, calc_distribution) — these describe
what a lesion LOOKS LIKE, not a diagnostic verdict about it, though it's
worth noting in your report that some of these (e.g. "irregular" shape,
"spiculated" margins) are themselves part of the BI-RADS suspicion lexicon
and carry real signal for that reason, not because they're neutral metadata.

Real data:
    python scripts/run_phase13_multimodal.py --epochs 30

Demo mode (synthetic data, proves the full pipeline runs end-to-end):
    python scripts/run_phase13_multimodal.py --demo --epochs 2
"""

import os
import sys
import argparse
import copy
import csv
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import load_config, set_global_seed, REPO_ROOT
from src.utils.metrics import compute_classification_metrics, format_metrics_report
from src.utils.class_weights import compute_class_weights
from src.utils.calibration import plot_reliability_diagram
from src.data.tabular_preprocessing import TabularPreprocessor
from src.data.multimodal_dataset import MultimodalDataset
from src.models.multimodal_model import build_multimodal_model

# 'assessment' is deliberately NOT in this list — see the module docstring.
CATEGORICAL_COLS = ["laterality", "view", "finding_type",
                    "mass_shape", "mass_margins", "calc_type", "calc_distribution"]
NUMERIC_COLS = ["breast_density", "subtlety"]


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, tabular, labels in loader:
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    return total_loss / len(loader.dataset), np.array(all_labels), np.array(all_probs)


def get_best_result(phase_name):
    log_path = os.path.join(REPO_ROOT, "reports", "eval_results", "experiments_log.csv")
    if not os.path.exists(log_path):
        return None
    df = pd.read_csv(log_path)
    rows = df[df["phase"] == phase_name].copy()
    if len(rows) == 0:
        return None
    rows["val_auc"] = pd.to_numeric(rows["val_auc"], errors="coerce")
    best = rows.loc[rows["val_auc"].idxmax()]
    return {"val_auc": best["val_auc"], "run_id": best["run_id"]}


def append_experiment_log(run_id, backbone, seed, val_auc, val_pr_auc, notes):
    log_path = os.path.join(REPO_ROOT, "reports", "eval_results", "experiments_log.csv")
    row = {
        "run_id": run_id, "date": datetime.now().isoformat(timespec="seconds"),
        "phase": "13_multimodal", "model": backbone, "dataset_split": "train/val",
        "seed": seed, "config_notes": notes,
        "val_auc": f"{val_auc:.4f}" if val_auc is not None else "",
        "val_pr_auc": f"{val_pr_auc:.4f}" if val_pr_auc is not None else "",
        "test_auc": "", "test_pr_auc": "", "notes": "",
    }
    file_exists = os.path.exists(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def generate_demo_data(n=24, seed=42):
    import cv2
    from src.data.image_io import generate_synthetic_mammogram
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        img = generate_synthetic_mammogram(size=300, seed=i)
        path = f"/tmp/mammotwin_phase13_demo_{i}.jpg"
        cv2.imwrite(path, img)
        finding_type = rng.choice(["mass", "calc"])
        rows.append({
            "image_id": f"demo_{i}", "image_file_path_resolved": path,
            "pathology_binary": rng.choice(["benign", "malignant"]),
            "laterality": rng.choice(["LEFT", "RIGHT"]),
            "view": rng.choice(["CC", "MLO"]),
            "finding_type": finding_type,
            "breast_density": rng.choice([1, 2, 3, 4, np.nan]),
            "subtlety": rng.choice([1, 2, 3, 4, 5]),
            "mass_shape": rng.choice(["OVAL", "IRREGULAR", None]) if finding_type == "mass" else None,
            "mass_margins": rng.choice(["CIRCUMSCRIBED", "SPICULATED", None]) if finding_type == "mass" else None,
            "calc_type": rng.choice(["PLEOMORPHIC", "PUNCTATE", None]) if finding_type == "calc" else None,
            "calc_distribution": rng.choice(["CLUSTERED", "LINEAR", None]) if finding_type == "calc" else None,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 13: multimodal fusion")
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    seed = config["project"]["seed"]
    set_global_seed(seed)
    torch.manual_seed(seed)

    epochs = args.epochs or config["training"]["num_epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = args.lr or config["training"]["learning_rate"]
    patience = config["training"].get("early_stopping_patience", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Structured features used: {CATEGORICAL_COLS + NUMERIC_COLS}")
    print("NOTE: 'assessment' (BI-RADS score) is deliberately EXCLUDED — "
          "it is the radiologist's own near-diagnosis and would leak the label.\n")

    if args.demo:
        print("Running Phase 13 in DEMO mode on synthetic data.\n")
        train_df = generate_demo_data(n=24, seed=seed)
        val_df = generate_demo_data(n=8, seed=seed + 1)
    else:
        metadata_dir = config["paths"]["data_metadata"]
        train_csv = args.train_csv or os.path.join(metadata_dir, "train_split.csv")
        val_csv = args.val_csv or os.path.join(metadata_dir, "val_split.csv")
        if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
            print(f"Could not find {train_csv} / {val_csv}. Run Phase 3 first.")
            return
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

    available_cat = [c for c in CATEGORICAL_COLS if c in train_df.columns]
    available_num = [c for c in NUMERIC_COLS if c in train_df.columns]
    missing = set(CATEGORICAL_COLS + NUMERIC_COLS) - set(available_cat + available_num)
    if missing:
        print(f"WARNING: columns not found in data, skipping: {missing}")

    tabular_pp = TabularPreprocessor(available_cat, available_num)
    tabular_pp.fit(train_df)  # fit on TRAIN only
    print(f"Tabular feature vector size: {tabular_pp.output_dim}\n")

    train_dataset = MultimodalDataset(train_df, config, tabular_pp)
    val_dataset = MultimodalDataset(val_df, config, tabular_pp)
    print(f"Train: {len(train_dataset)} images, class counts: {train_dataset.class_counts()}")
    print(f"Val:   {len(val_dataset)} images, class counts: {val_dataset.class_counts()}\n")

    if len(train_dataset) < 4 or len(val_dataset) < 2:
        print("Too few examples to train/evaluate.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, persistent_workers=(args.num_workers > 0))

    model = build_multimodal_model(config, tabular_input_dim=tabular_pp.output_dim).to(device)
    class_weights = compute_class_weights(train_dataset.class_counts(), config["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=config["training"].get("weight_decay", 0.0))

    models_dir = config["paths"]["models_dir"]
    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    run_id = f"multimodal_{config['model']['backbone']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_path = os.path.join(models_dir, f"{run_id}.pt")

    best_val_auc = -1.0
    best_val_labels, best_val_probs = None, None
    epochs_without_improvement = 0

    print(f"Training for up to {epochs} epochs (early stopping patience: {patience})...\n")
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        val_auc = None
        if len(np.unique(val_labels)) > 1:
            from src.utils.stats import roc_auc_score
            val_auc = roc_auc_score(val_labels, val_probs)

        auc_str = f"{val_auc:.4f}" if val_auc is not None else "N/A"
        print(f"Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | val_auc: {auc_str}")

        if val_auc is not None and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_labels, best_val_probs = val_labels.copy(), val_probs.copy()
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "tabular_categorical_cols": available_cat, "tabular_numeric_cols": available_num,
                        "epoch": epoch, "val_auc": val_auc}, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val AUC: {best_val_auc:.4f} (checkpoint saved: {checkpoint_path})")

    from src.utils.stats import average_precision_score
    best_val_pr_auc = average_precision_score(best_val_labels, best_val_probs)
    metrics = compute_classification_metrics(best_val_labels, best_val_probs)
    print("\n" + format_metrics_report(metrics))

    calib_path = os.path.join(figures_dir, "phase13_calibration.png")
    calib_metrics = plot_reliability_diagram(best_val_labels, best_val_probs, calib_path,
                                              title="Multimodal Classifier Calibration")
    print(f"\nBrier score: {calib_metrics['brier_score']:.4f}")

    registry_path = os.path.join(models_dir, "registry.json")
    registry = {}
    if os.path.exists(registry_path) and os.path.getsize(registry_path) > 0:
        try:
            with open(registry_path) as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            registry = {}
    registry[run_id] = {
        "checkpoint_path": checkpoint_path, "backbone": config["model"]["backbone"],
        "trained_date": datetime.now().isoformat(timespec="seconds"),
        "val_auc": best_val_auc, "val_pr_auc": best_val_pr_auc, "seed": seed,
        "phase": "13_multimodal",
    }
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    append_experiment_log(run_id, config["model"]["backbone"], seed, best_val_auc, best_val_pr_auc,
                           notes=f"epochs={epochs},batch_size={batch_size},lr={lr}")

    print("\n" + "=" * 60)
    print("WHOLE-IMAGE ONLY (Phase 6) vs. MULTIMODAL (Phase 13)")
    print("=" * 60)
    phase6_result = get_best_result("6_baseline")
    if phase6_result:
        print(f"Whole-image-only val AUC: {phase6_result['val_auc']:.4f}  (run: {phase6_result['run_id']})")
        print(f"Multimodal val AUC:       {best_val_auc:.4f}  (run: {run_id})")
        diff = best_val_auc - phase6_result["val_auc"]
        print(f"\nMultimodal is {abs(diff):.4f} AUC {'HIGHER' if diff > 0 else 'LOWER'} than whole-image-only.")

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.bar(["Whole-image only\n(Phase 6)", "Multimodal\n(Phase 13)"],
               [phase6_result["val_auc"], best_val_auc], color=["#4C72B0", "#55A868"])
        ax.set_ylabel("Validation ROC-AUC")
        ax.set_ylim([0, 1])
        ax.set_title("Whole-image vs. Multimodal Fusion")
        plt.tight_layout()
        comparison_path = os.path.join(figures_dir, "phase13_comparison.png")
        plt.savefig(comparison_path, dpi=150)
        print(f"Saved comparison chart: {comparison_path}")
    else:
        print("No logged Phase 6 result found to compare against.")

    print("\nPhase 13 multimodal model COMPLETE.")


if __name__ == "__main__":
    main()