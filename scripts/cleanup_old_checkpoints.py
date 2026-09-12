"""
Deletes old/invalid segmentation and detector checkpoints, keeping only
the ones you specify, and removes their corresponding entries from
models/registry.json so future auto-discovery (find_best_checkpoint)
can't accidentally pick a stale or invalid run again.

Review the KEEP_FILES list below before running -- this permanently
deletes files.

Usage:
    python cleanup_old_checkpoints.py            # dry run, lists what WOULD be deleted
    python cleanup_old_checkpoints.py --confirm  # actually deletes
"""
import os
import sys
import json
import glob

MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")

# Checkpoints to KEEP (everything else matching these prefixes gets removed).
KEEP_FILES = {
    "segmentation_unet_20260910_214314.pt",  # full-dataset, patch-256, max-pos-weight-30 run
    "detector_fasterrcnn_20260904_113843.pt",  # the only correctly-registered detector run
}

# Only prune files matching these prefixes -- never touches baseline/lesion_crop/multimodal.
PRUNE_PREFIXES = ("segmentation_unet_", "detector_fasterrcnn_")


def main():
    confirm = "--confirm" in sys.argv

    candidates = []
    for prefix in PRUNE_PREFIXES:
        candidates.extend(glob.glob(os.path.join(MODELS_DIR, f"{prefix}*.pt")))

    to_delete = [p for p in candidates if os.path.basename(p) not in KEEP_FILES]

    if not to_delete:
        print("Nothing to delete -- all matching checkpoints are in KEEP_FILES.")
        return

    print("The following files will be deleted:" if confirm else
          "DRY RUN -- the following files WOULD be deleted (pass --confirm to actually delete):")
    for p in to_delete:
        print(f"  {p}")

    if not confirm:
        print("\nNo changes made. Re-run with --confirm to apply.")
        return

    for p in to_delete:
        os.remove(p)
    print(f"\nDeleted {len(to_delete)} file(s).")

    # Clean matching entries out of registry.json so auto-discovery never
    # points at a now-missing (or previously-invalid) checkpoint again.
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        deleted_paths = {os.path.normpath(p) for p in to_delete}
        before = len(registry)
        registry = {k: v for k, v in registry.items()
                    if os.path.normpath(v.get("checkpoint_path", "")) not in deleted_paths}
        removed = before - len(registry)
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"Removed {removed} stale entr{'y' if removed == 1 else 'ies'} from {REGISTRY_PATH}.")


if __name__ == "__main__":
    main()
