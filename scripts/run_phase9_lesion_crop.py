"""
Phase 9 deliverable: train a lesion-CROP classifier (using Phase 7's real
bounding boxes) and compare it against Phase 6's whole-image baseline.

Prerequisite: bbox_metadata_train.csv / bbox_metadata_val.csv from Phase 7.
If those files don't already have a pathology_binary column (older runs,
before this script existed), it's merged in automatically from
train_split.csv / val_split.csv — no need to re-run Phase 7's slow
bbox-extraction step.

Real data:
    python scripts/run_phase9_lesion_crop.py --epochs 20

Demo mode (synthetic data, proves the full pipeline runs end-to-end):
    python scripts/run_phase9_lesion_crop.py --demo --epochs 2
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
from src.data.crop_dataset import LesionCropDataset
from src.models.classifier import build_classifier


def attach_labels_if_missing(bbox_df, split_csv_path, label_col="pathology_binary"):
    if label_col in bbox_df.columns and bbox_df[label_col].notna().any():
        return bbox_df
    if not os.path.exists(split_csv_path):
        print(f"WARNING: {label_col} missing from bbox metadata and {split_csv_path} "
              f"not found to merge from.")
        return bbox_df
    split_df = pd.read_csv(split_csv_path)[["image_id", label_col]]
    merged = bbox_df.drop(columns=[label_col], errors="ignore").merge(split_df, on="image_id", how="left")
    print(f"Merged '{label_col}' into bbox metadata from {split_csv_path}.")
    return merged


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    return total_loss / len(loader.dataset), np.array(all_labels), np.array(all_probs)


def get_best_phase6_result():
    """Read the experiments log for the best logged Phase 6 whole-image
    baseline result, to compare against."""
    log_path = os.path.join(REPO_ROOT, "reports", "eval_results", "experiments_log.csv")
    if not os.path.exists(log_path):
        return None
    df = pd.read_csv(log_path)
    phase6_rows = df[df["phase"] == "6_baseline"]
    if len(phase6_rows) == 0:
        return None
    phase6_rows = phase6_rows.copy()
    phase6_rows["val_auc"] = pd.to_numeric(phase6_rows["val_auc"], errors="coerce")
    best_row = phase6_rows.loc[phase6_rows["val_auc"].idxmax()]
    return {"val_auc": best_row["val_auc"], "val_pr_auc": pd.to_numeric(best_row["val_pr_auc"], errors="coerce"),
            "run_id": best_row["run_id"]}


def append_experiment_log(run_id, backbone, seed, val_auc, val_pr_auc, notes):
    log_path = os.path.join(REPO_ROOT, "reports", "eval_results", "experiments_log.csv")
    row = {
        "run_id": run_id, "date": datetime.now().isoformat(timespec="seconds"),
        "phase": "9_lesion_crop", "model": backbone, "dataset_split": "train/val",
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


def generate_demo_dataset(config, n=20, seed=42):
    sys.path.insert(0, os.path.dirname(__file__))
    from build_bbox_metadata import generate_demo_rows, process_row

    df, raw_images_dir = generate_demo_rows(n=n, seed=seed)
    results = []
    for _, row in df.iterrows():
        result = process_row(row, raw_images_dir)
        result["image_id"] = row["image_id"]
        results.append(result)
    result_df = pd.DataFrame(results)
    return result_df[result_df["has_bbox"] == True].reset_index(drop=True)  # noqa: E712


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 9: lesion-crop classifier")
    parser.add_argument("--train-bbox-csv", type=str, default=None)
    parser.add_argument("--val-bbox-csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--crop-padding", type=float, default=0.2)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    if args.image_size:
        config = copy.deepcopy(config)
        config["preprocessing"]["image_size"] = [args.image_size, args.image_size]
        print(f"Overriding image_size to {args.image_size}x{args.image_size}.\n")

    seed = config["project"]["seed"]
    set_global_seed(seed)
    torch.manual_seed(seed)

    epochs = args.epochs or config["training"]["num_epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = args.lr or config["training"]["learning_rate"]
    patience = config["training"].get("early_stopping_patience", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.demo:
        print("Running Phase 9 in DEMO mode on synthetic data.\n")
        train_df = generate_demo_dataset(config, n=24, seed=seed)
        val_df = generate_demo_dataset(config, n=8, seed=seed + 1)
    else:
        metadata_dir = config["paths"]["data_metadata"]
        train_csv = args.train_bbox_csv or os.path.join(metadata_dir, "bbox_metadata_train.csv")
        val_csv = args.val_bbox_csv or os.path.join(metadata_dir, "bbox_metadata_val.csv")
        if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
            print(f"Could not find {train_csv} / {val_csv}. Run Phase 7's build_bbox_metadata.py first.")
            return
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        train_df = attach_labels_if_missing(train_df, os.path.join(metadata_dir, "train_split.csv"))
        val_df = attach_labels_if_missing(val_df, os.path.join(metadata_dir, "val_split.csv"))

    train_dataset = LesionCropDataset(train_df, config, crop_padding_fraction=args.crop_padding)
    val_dataset = LesionCropDataset(val_df, config, crop_padding_fraction=args.crop_padding)
    print(f"Train: {len(train_dataset)} crops, class counts: {train_dataset.class_counts()}")
    print(f"Val:   {len(val_dataset)} crops, class counts: {val_dataset.class_counts()}\n")

    if len(train_dataset) < 4 or len(val_dataset) < 2:
        print("Too few labeled crops to train/evaluate. Check the label-merge step above, "
              "or run build_bbox_metadata.py with a higher --limit.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, persistent_workers=(args.num_workers > 0))

    model = build_classifier(config).to(device)
    class_weights = compute_class_weights(train_dataset.class_counts(), config["model"]["num_classes"]).to(device)
    print(f"Class weights (inverse frequency): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=config["training"].get("weight_decay", 0.0))

    models_dir = config["paths"]["models_dir"]
    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    run_id = f"lesion_crop_{config['model']['backbone']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_path = os.path.join(models_dir, f"{run_id}.pt")

    best_val_auc = -1.0
    best_val_labels, best_val_probs = None, None
    epochs_without_improvement = 0

    print(f"\nTraining for up to {epochs} epochs (early stopping patience: {patience})...\n")
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        val_auc = None
        if len(np.unique(val_labels)) > 1:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(val_labels, val_probs)

        auc_str = f"{val_auc:.4f}" if val_auc is not None else "N/A"
        print(f"Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | val_auc: {auc_str}")

        if val_auc is not None and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_labels, best_val_probs = val_labels.copy(), val_probs.copy()
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "val_auc": val_auc}, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val AUC: {best_val_auc:.4f} (checkpoint saved: {checkpoint_path})")

    from sklearn.metrics import average_precision_score
    best_val_pr_auc = average_precision_score(best_val_labels, best_val_probs)
    metrics = compute_classification_metrics(best_val_labels, best_val_probs)
    print("\n" + format_metrics_report(metrics))

    # --- Calibration ---
    calib_path = os.path.join(figures_dir, "phase9_calibration.png")
    calib_metrics = plot_reliability_diagram(best_val_labels, best_val_probs, calib_path,
                                              title="Lesion-Crop Classifier Calibration")
    print(f"\nBrier score: {calib_metrics['brier_score']:.4f}")
    print(f"Saved reliability diagram: {calib_path}")

    # --- Update registry + experiments log ---
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
        "phase": "9_lesion_crop",
    }
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    append_experiment_log(run_id, config["model"]["backbone"], seed, best_val_auc, best_val_pr_auc,
                           notes=f"epochs={epochs},batch_size={batch_size},lr={lr},crop_padding={args.crop_padding}")

    # --- Compare against Phase 6 whole-image baseline ---
    phase6_result = get_best_phase6_result()
    print("\n" + "=" * 60)
    print("WHOLE-IMAGE (Phase 6) vs. LESION-CROP (Phase 9)")
    print("=" * 60)
    if phase6_result:
        print(f"Whole-image val AUC: {phase6_result['val_auc']:.4f}  (run: {phase6_result['run_id']})")
        print(f"Lesion-crop val AUC: {best_val_auc:.4f}  (run: {run_id})")
        diff = best_val_auc - phase6_result["val_auc"]
        direction = "HIGHER" if diff > 0 else "LOWER"
        print(f"\nLesion-crop is {abs(diff):.4f} AUC {direction} than whole-image.")

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.bar(["Whole-image\n(Phase 6)", "Lesion-crop\n(Phase 9)"],
               [phase6_result["val_auc"], best_val_auc], color=["#4C72B0", "#C44E52"])
        ax.set_ylabel("Validation ROC-AUC")
        ax.set_ylim([0, 1])
        ax.set_title("Whole-image vs. Lesion-crop Classification")
        plt.tight_layout()
        comparison_path = os.path.join(figures_dir, "phase9_comparison.png")
        plt.savefig(comparison_path, dpi=150)
        print(f"Saved comparison chart: {comparison_path}")
    else:
        print("No logged Phase 6 result found in experiments_log.csv to compare against.")
        print("Run Phase 6 first for a meaningful whole-image-vs-crop comparison.")

    print("\nPhase 9 lesion-crop classifier COMPLETE.")


if __name__ == "__main__":
    main()