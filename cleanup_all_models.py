"""
Deletes every checkpoint EXCEPT the final ones chosen for this project,
and removes their corresponding entries from models/registry.json so
future auto-discovery never picks up a deleted file.

Review the KEEP_FILES list below before running -- this permanently
deletes files. Always dry-run first (the default).

Usage:
    python cleanup_all_models.py            # dry run -- shows what WOULD be deleted
    python cleanup_all_models.py --confirm  # actually deletes
"""
import os
import sys
import json
import glob

MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")

# The final, chosen checkpoint for each model type -- everything else
# matching these prefixes gets deleted.
KEEP_FILES = {
    "baseline_resnet50_20260913_175719.pt",
    "lesion_crop_resnet50_20260919_095210.pt",
    "multimodal_resnet50_20260913_192536.pt",
    "detector_fasterrcnn_20260914_162708.pt",
    "segmentation_unet_20260919_101818.pt",
}

PRUNE_PREFIXES = (
    "baseline_",
    "lesion_crop_",
    "multimodal_",
    "detector_fasterrcnn_",
    "segmentation_unet_",
)


def main():
    confirm = "--confirm" in sys.argv

    candidates = []
    for prefix in PRUNE_PREFIXES:
        candidates.extend(glob.glob(os.path.join(MODELS_DIR, f"{prefix}*.pt")))

    to_delete = [p for p in candidates if os.path.basename(p) not in KEEP_FILES]
    to_keep_found = [p for p in candidates if os.path.basename(p) in KEEP_FILES]

    print("Files that will be KEPT:")
    for p in sorted(to_keep_found):
        print(f"  {p}")
    missing_keeps = KEEP_FILES - {os.path.basename(p) for p in to_keep_found}
    if missing_keeps:
        print("\nWARNING: the following KEEP_FILES were not found in the models folder "
              "(check for typos in filenames, or that they weren't already deleted):")
        for f in missing_keeps:
            print(f"  {f}")

    if not to_delete:
        print("\nNothing to delete -- all matching checkpoints are already just the keepers.")
        return

    print(f"\n{'Deleting' if confirm else 'DRY RUN -- would delete'} {len(to_delete)} file(s):")
    total_size = 0
    for p in to_delete:
        size = os.path.getsize(p)
        total_size += size
        print(f"  {p}  ({size / 1e6:.1f} MB)")
    print(f"\nTotal space {'freed' if confirm else 'that would be freed'}: {total_size / 1e6:.1f} MB")

    if not confirm:
        print("\nNo changes made. Re-run with --confirm to apply.")
        return

    for p in to_delete:
        os.remove(p)
    print(f"\nDeleted {len(to_delete)} file(s).")

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
        print(f"\nRemaining registry entries: {list(registry.keys())}")


if __name__ == "__main__":
    main()