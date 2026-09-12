"""
One-off fix: adds the two correct, already-trained checkpoints back into
models/registry.json, using the exact metrics printed during their
original training runs. This is NOT a re-run or re-training -- it just
restores the bookkeeping entry so find_best_checkpoint() can auto-discover
these checkpoints again (no more typing --segmentation-checkpoint /
--detector-checkpoint by hand every time).

Safe to run multiple times -- it overwrites only these two specific keys.
"""
import os
import json
from datetime import datetime

MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")

ENTRIES = {
    "detector_fasterrcnn_20260904_113843": {
        "checkpoint_path": os.path.join(MODELS_DIR, "detector_fasterrcnn_20260904_113843.pt"),
        "trained_date": "2026-09-04T11:38:43",
        "val_map": 0.0444, "val_map_50": 0.1200, "val_mean_top1_iou": 0.1809,
        "seed": 42, "phase": "7_localization",
    },
    "segmentation_unet_20260910_214314": {
        "checkpoint_path": os.path.join(MODELS_DIR, "segmentation_unet_20260910_214314.pt"),
        "trained_date": "2026-09-10T21:43:14",
        "val_dice": 0.1561, "seed": 42, "patch_size": 256,
        "phase": "8_segmentation",
    },
}


def main():
    registry = {}
    if os.path.exists(REGISTRY_PATH) and os.path.getsize(REGISTRY_PATH) > 0:
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)

    for run_id, entry in ENTRIES.items():
        if not os.path.exists(entry["checkpoint_path"]):
            print(f"SKIPPED {run_id}: checkpoint file not found at {entry['checkpoint_path']} "
                  f"-- did cleanup delete it? Not adding a registry entry for a missing file.")
            continue
        registry[run_id] = entry
        print(f"Added/updated: {run_id}")

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\nSaved {REGISTRY_PATH}. find_best_checkpoint() should now auto-discover "
          f"these without needing explicit --checkpoint flags.")


if __name__ == "__main__":
    main()
