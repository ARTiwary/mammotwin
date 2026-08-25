"""
Phase 6 deliverable: train and evaluate the baseline classifier.

Real data:
    python scripts/run_phase6_baseline.py --epochs 30

Quick smoke test on a small slice of real data (useful before committing
to a full run):
    python scripts/run_phase6_baseline.py --epochs 2 --limit 100

Demo mode (synthetic data, proves the whole pipeline runs end-to-end):
    python scripts/run_phase6_baseline.py --demo --epochs 2
"""

import os
import sys
import argparse
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

from src.utils.config import load_config, set_global_seed
from src.utils.metrics import compute_classification_metrics, format_metrics_report
from src.data.dataset import MammogramDataset, LABEL_MAP
from src.models.classifier import build_classifier


def generate_synthetic_split_csvs(n_train=24, n_val=8, seed=42):
    """Small synthetic train/val metadata + matching images, purely to
    smoke-test the full training loop before running on real data."""
    import cv2
    from src.data.image_io import generate_synthetic_mammogram

    rng = np.random.default_rng(seed)
    tmp_dir = "/tmp/mammotwin_phase6_demo_images"
    os.makedirs(tmp_dir, exist_ok=True)

    def make_split(n, offset):
        rows = []
        for i in range(n):
            idx = offset + i
            pathology = rng.choice(["benign", "malignant"])
            img_path = os.path.join(tmp_dir, f"img_{idx}.jpg")
            img = generate_synthetic_mammogram(size=300, seed=idx)
            cv2.imwrite(img_path, img)
            rows.append({
                "patient_id": f"P_{idx:04d}",
                "pathology_binary": pathology,
                "image_file_path_resolved": img_path,
            })
        return pd.DataFrame(rows)

    return make_split(n_train, 0), make_split(n_val, n_train)


def compute_class_weights(dataset: MammogramDataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weighting for CrossEntropyLoss, so the
    majority class doesn't dominate the loss on an imbalanced dataset."""
    counts = dataset.class_counts()  # e.g. {'benign': 1470, 'malignant': 1031}
    total = sum(counts.values())
    weights = torch.ones(num_classes)
    for label_name, label_idx in LABEL_MAP.items():
        count = counts.get(label_name, 0)
        if count > 0:
            weights[label_idx] = total / (num_classes * count)
    return weights


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
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(malignant)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_probs)


def append_experiment_log(config, run_id, backbone, seed, best_val_auc, best_val_pr_auc, notes):
    from src.utils.config import REPO_ROOT
    log_path = os.path.join(REPO_ROOT, "reports", "eval_results", "experiments_log.csv")

    row = {
        "run_id": run_id, "date": datetime.now().isoformat(timespec="seconds"),
        "phase": "6_baseline", "model": backbone, "dataset_split": "train/val",
        "seed": seed, "config_notes": notes,
        "val_auc": f"{best_val_auc:.4f}" if best_val_auc is not None else "",
        "val_pr_auc": f"{best_val_pr_auc:.4f}" if best_val_pr_auc is not None else "",
        "test_auc": "", "test_pr_auc": "", "notes": "",
    }
    file_exists = os.path.exists(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 6: baseline classifier")
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap on training rows, for a quick smoke test on real data")
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

    # --- Load data ---
    if args.demo:
        print("Running Phase 6 in DEMO mode on synthetic data.\n")
        train_df, val_df = generate_synthetic_split_csvs(seed=seed)
    else:
        metadata_dir = config["paths"]["data_metadata"]
        train_csv = args.train_csv or os.path.join(metadata_dir, "train_split.csv")
        val_csv = args.val_csv or os.path.join(metadata_dir, "val_split.csv")
        if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
            print(f"Could not find {train_csv} / {val_csv}. Run Phase 3 first, or pass --demo.")
            return
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        if args.limit:
            train_df = train_df.head(args.limit)
            print(f"--limit set: using only {len(train_df)} training rows for a quick run.")

    train_dataset = MammogramDataset(train_df, config)
    val_dataset = MammogramDataset(val_df, config)
    print(f"Train: {len(train_dataset)} images, class counts: {train_dataset.class_counts()}")
    print(f"Val:   {len(val_dataset)} images, class counts: {val_dataset.class_counts()}\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=args.num_workers,
                               persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             persistent_workers=(args.num_workers > 0))

    # --- Model, loss, optimizer ---
    model = build_classifier(config).to(device)
    class_weights = compute_class_weights(train_dataset, config["model"]["num_classes"]).to(device)
    print(f"Class weights (inverse frequency): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=config["training"].get("weight_decay", 0.0))

    # --- Training loop ---
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_auc = -1.0
    best_val_pr_auc = None
    best_val_labels, best_val_probs = None, None
    epochs_without_improvement = 0

    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    run_id = f"baseline_{config['model']['backbone']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_path = os.path.join(models_dir, f"{run_id}.pt")

    print(f"\nTraining for up to {epochs} epochs (early stopping patience: {patience})...\n")
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        val_auc = None
        if len(np.unique(val_labels)) > 1:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(val_labels, val_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        auc_str = f"{val_auc:.4f}" if val_auc is not None else "N/A"
        print(f"Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | val_auc: {auc_str}")

        if val_auc is not None and val_auc > best_val_auc:
            best_val_auc = val_auc
            from sklearn.metrics import average_precision_score
            best_val_pr_auc = average_precision_score(val_labels, val_probs)
            # Cache these NOW, while we have them, instead of reloading the
            # checkpoint and re-running the val DataLoader afterward — that
            # extra pass is what was crashing on Windows (multiprocessing
            # worker spawn/teardown instability across repeated iterations).
            best_val_labels, best_val_probs = val_labels.copy(), val_probs.copy()
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config, "epoch": epoch, "val_auc": val_auc,
            }, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    print(f"\nBest val AUC: {best_val_auc:.4f} (checkpoint saved: {checkpoint_path})")

    # --- Final evaluation report, using CACHED predictions from the best
    # epoch (no need to reload the checkpoint or re-run the val DataLoader) ---
    metrics = compute_classification_metrics(best_val_labels, best_val_probs)
    print("\n" + format_metrics_report(metrics))

    # --- Save training curves ---
    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[0].set_title("Loss")
    valid_aucs = [a for a in history["val_auc"] if a is not None]
    if valid_aucs:
        axes[1].plot(valid_aucs)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val ROC-AUC")
        axes[1].set_title("Validation ROC-AUC")
    plt.tight_layout()
    curves_path = os.path.join(figures_dir, "phase6_training_curves.png")
    plt.savefig(curves_path, dpi=150)
    print(f"\nSaved training curves: {curves_path}")

    # --- Update model registry ---
    registry_path = os.path.join(models_dir, "registry.json")
    registry = {}
    if os.path.exists(registry_path) and os.path.getsize(registry_path) > 0:
        try:
            with open(registry_path) as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: {registry_path} exists but isn't valid JSON — "
                  f"starting a fresh registry (old file left untouched on disk).")
            registry = {}
    registry[run_id] = {
        "checkpoint_path": checkpoint_path, "backbone": config["model"]["backbone"],
        "trained_date": datetime.now().isoformat(timespec="seconds"),
        "val_auc": best_val_auc, "val_pr_auc": best_val_pr_auc, "seed": seed,
    }
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Updated model registry: {registry_path}")

    # --- Append to experiments log ---
    append_experiment_log(config, run_id, config["model"]["backbone"], seed,
                           best_val_auc, best_val_pr_auc,
                           notes=f"epochs={epochs},batch_size={batch_size},lr={lr}")
    print("Logged run to reports/eval_results/experiments_log.csv")

    print("\nPhase 6 baseline classifier COMPLETE.")
    print("Do NOT touch the test set yet — that happens once, in Phase 14.")


if __name__ == "__main__":
    main()