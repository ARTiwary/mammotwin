"""
Identifies which checkpoint in models/ is the actual "keeper" for each
model type, and lists everything else as safe to delete — freeing disk
space from superseded training runs.

Keeper logic:
  - Classifiers (baseline, lesion_crop, multimodal): the checkpoint with
    the HIGHEST val_auc logged in registry.json for that phase.
  - Detector / segmentation: registry.json doesn't log these (their
    scripts never write to it), so the MOST RECENT (by filename timestamp)
    checkpoint is kept as a reasonable default — override with
    --keep-detector / --keep-segmentation if you know a specific one
    performed better.

SAFE BY DEFAULT: only prints what it WOULD delete. Nothing is actually
removed unless you pass --confirm-delete.

Usage:
    python scripts/cleanup_old_checkpoints.py
    python scripts/cleanup_old_checkpoints.py --confirm-delete
"""

import os
import sys
import re
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-delete", action="store_true",
                         help="Actually delete the files. Without this, only prints the plan.")
    parser.add_argument("--keep-detector", type=str, default=None,
                         help="Explicit detector checkpoint filename to keep (default: most recent)")
    parser.add_argument("--keep-segmentation", type=str, default=None,
                         help="Explicit segmentation checkpoint filename to keep (default: most recent)")
    args = parser.parse_args()

    config = load_config()
    models_dir = config["paths"]["models_dir"]
    registry_path = os.path.join(models_dir, "registry.json")

    registry = {}
    if os.path.exists(registry_path) and os.path.getsize(registry_path) > 0:
        with open(registry_path) as f:
            registry = json.load(f)

    all_checkpoints = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
    if not all_checkpoints:
        print("No checkpoints found in", models_dir)
        return

    keepers = set()

    # --- Classifiers: best val_auc per phase, using registry + prefix fallback ---
    phase_prefixes = {
        "6_baseline": "baseline_",
        "9_lesion_crop": "lesion_crop_",
        "13_multimodal": "multimodal_",
    }
    for phase, prefix in phase_prefixes.items():
        matching = {k: v for k, v in registry.items() if v.get("phase") == phase}
        if not matching:
            matching = {k: v for k, v in registry.items() if k.startswith(prefix)}
        if matching:
            best_key = max(matching, key=lambda k: matching[k].get("val_auc", -1))
            checkpoint_path = matching[best_key]["checkpoint_path"]
            keepers.add(os.path.basename(checkpoint_path))
            print(f"Keeper for {phase}: {os.path.basename(checkpoint_path)} "
                  f"(val_auc={matching[best_key].get('val_auc', '?')})")

    # --- Detector / segmentation: most recent by filename timestamp, or explicit override ---
    def most_recent(prefix):
        candidates = sorted([f for f in all_checkpoints if f.startswith(prefix)])
        return candidates[-1] if candidates else None  # timestamps sort lexicographically

    detector_keeper = args.keep_detector or most_recent("detector_")
    segmentation_keeper = args.keep_segmentation or most_recent("segmentation_")
    if detector_keeper:
        keepers.add(detector_keeper)
        print(f"Keeper for detector: {detector_keeper} (most recent)")
    if segmentation_keeper:
        keepers.add(segmentation_keeper)
        print(f"Keeper for segmentation: {segmentation_keeper} (most recent)")

    to_delete = [f for f in all_checkpoints if f not in keepers]

    print(f"\n{'=' * 60}")
    print(f"KEEPING {len(keepers)} checkpoints, DELETING {len(to_delete)}:")
    print(f"{'=' * 60}")
    total_size_mb = 0
    for f in sorted(to_delete):
        path = os.path.join(models_dir, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        total_size_mb += size_mb
        print(f"  {f}  ({size_mb:.1f} MB)")

    print(f"\nTotal space to free: {total_size_mb:.1f} MB")

    if not args.confirm_delete:
        print("\nDRY RUN — nothing deleted. Re-run with --confirm-delete to actually remove these files.")
        return

    for f in to_delete:
        os.remove(os.path.join(models_dir, f))
    print(f"\nDeleted {len(to_delete)} files, freed {total_size_mb:.1f} MB.")


if __name__ == "__main__":
    main()