"""
Phase 8 deliverable: train and evaluate the U-Net lesion segmentation model.

Prerequisite: bbox_metadata_train.csv / bbox_metadata_val.csv from Phase 7
(they already contain the resolved mask path per row — no new ground-truth
step needed).

Real data:
    python scripts/run_phase8_segmentation.py --epochs 20

Detection/segmentation both benefit from higher resolution than the
classifier's default — override independently if needed:
    python scripts/run_phase8_segmentation.py --epochs 20 --image-size 384

Demo mode (synthetic data, proves the full pipeline runs end-to-end):
    python scripts/run_phase8_segmentation.py --demo --epochs 2
"""

import os
import sys
import argparse
import copy
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, set_global_seed
from src.utils.segmentation_metrics import DiceBCELoss, compute_segmentation_metrics
from src.data.segmentation_dataset import LesionSegmentationDataset
from src.models.segmentation_model import build_segmentation_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_examples: int = 4):
    model.eval()
    total_loss = 0.0
    all_dice, all_iou = [], []
    examples = []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        batch_metrics = compute_segmentation_metrics(logits, masks)
        all_dice.append(batch_metrics["dice"])
        all_iou.append(batch_metrics["iou"])

        if len(examples) < n_examples:
            probs = torch.sigmoid(logits)
            for i in range(images.size(0)):
                if len(examples) >= n_examples:
                    break
                examples.append((
                    images[i, 0].cpu().numpy(),
                    masks[i, 0].cpu().numpy(),
                    probs[i, 0].cpu().numpy(),
                ))

    avg_loss = total_loss / len(loader.dataset)
    return {
        "loss": avg_loss,
        "dice": float(np.mean(all_dice)),
        "iou": float(np.mean(all_iou)),
    }, examples


def save_overlay_figure(examples, out_path):
    if not examples:
        return
    n = len(examples)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    for i, (img, gt_mask, pred_prob) in enumerate(examples):
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title("Input")
        axes[0, i].axis("off")

        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].imshow(gt_mask, cmap="Greens", alpha=0.5)
        axes[1, i].set_title("Ground truth mask")
        axes[1, i].axis("off")

        axes[2, i].imshow(img, cmap="gray")
        axes[2, i].imshow(pred_prob, cmap="Reds", alpha=0.5)
        axes[2, i].set_title("Predicted mask (prob.)")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_demo_dataset(config, n=20, seed=42):
    """Reuse Phase 7's bbox demo generator (it already builds matching
    full-image + full-size-mask pairs, exactly what segmentation needs)."""
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
    parser = argparse.ArgumentParser(description="MammoTwin Phase 8: lesion segmentation")
    parser.add_argument("--train-bbox-csv", type=str, default=None)
    parser.add_argument("--val-bbox-csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None,
                         help="Override preprocessing.image_size for THIS script only.")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-quality-filter", action="store_true")
    parser.add_argument("--quality-report", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    if args.image_size:
        config = copy.deepcopy(config)
        config["preprocessing"]["image_size"] = [args.image_size, args.image_size]
        print(f"Overriding image_size to {args.image_size}x{args.image_size}.\n")

    seed = config["project"]["seed"]
    set_global_seed(seed)
    torch.manual_seed(seed)

    batch_size = args.batch_size or config["segmentation"]["batch_size"]
    lr = args.lr or config["segmentation"]["learning_rate"]
    bce_weight = config["segmentation"].get("bce_weight", 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.demo:
        print("Running Phase 8 in DEMO mode on synthetic data.\n")
        train_bbox_df = generate_demo_dataset(config, n=20, seed=seed)
        val_bbox_df = generate_demo_dataset(config, n=6, seed=seed + 1)
    else:
        metadata_dir = config["paths"]["data_metadata"]
        train_csv = args.train_bbox_csv or os.path.join(metadata_dir, "bbox_metadata_train.csv")
        val_csv = args.val_bbox_csv or os.path.join(metadata_dir, "bbox_metadata_val.csv")
        if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
            print(f"Could not find {train_csv} / {val_csv}.")
            print("Run scripts/build_bbox_metadata.py --split train and --split val first, "
                  "or pass --demo.")
            return
        train_bbox_df = pd.read_csv(train_csv)
        val_bbox_df = pd.read_csv(val_csv)

        if not args.no_quality_filter:
            from src.data.quality_filter import filter_by_quality
            quality_report_path = args.quality_report or os.path.join(metadata_dir, "quality_report.csv")
            train_bbox_df = filter_by_quality(train_bbox_df, quality_report_path)

    train_dataset = LesionSegmentationDataset(train_bbox_df, config, augment=(not args.no_augment))
    val_dataset = LesionSegmentationDataset(val_bbox_df, config, augment=False)  # NEVER augment val/test
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} "
          f"(augmentation: {'ON' if not args.no_augment else 'OFF'})\n")

    if len(train_dataset) < 2 or len(val_dataset) < 1:
        print("Too few images with usable masks. Check bbox_metadata CSVs, "
              "or run build_bbox_metadata.py with a higher --limit.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=args.num_workers,
                               persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             persistent_workers=(args.num_workers > 0))

    model = build_segmentation_model(config).to(device)
    criterion = DiceBCELoss(bce_weight=bce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    models_dir = config["paths"]["models_dir"]
    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    run_id = f"segmentation_unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_path = os.path.join(models_dir, f"{run_id}.pt")

    best_val_dice = -1.0
    print(f"Training for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _ = evaluate(model, val_loader, criterion, device, n_examples=0)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"  -> Reducing learning rate: {prev_lr:.2e} -> {new_lr:.2e}")

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_metrics['loss']:.4f} | val_dice: {val_metrics['dice']:.4f} | "
              f"val_iou: {val_metrics['iou']:.4f}")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "val_dice": val_metrics["dice"]}, checkpoint_path)

    print(f"\nBest val Dice: {best_val_dice:.4f} (checkpoint saved: {checkpoint_path})")

    print("\nFinal evaluation with example visualizations...")
    final_metrics, examples = evaluate(model, val_loader, criterion, device, n_examples=4)
    print(f"Final val Dice: {final_metrics['dice']:.4f} | Final val IoU: {final_metrics['iou']:.4f}")

    fig_path = os.path.join(figures_dir, "phase8_segmentation_examples.png")
    save_overlay_figure(examples, fig_path)
    print(f"Saved overlay visualization: {fig_path}")

    print("\nPhase 8 segmentation COMPLETE.")


if __name__ == "__main__":
    main()