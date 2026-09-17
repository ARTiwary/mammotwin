"""
Diagnostic: bbox_metadata_train.csv reported 2445/2501 successful bbox
extractions, but Phase 8 (segmentation) and Phase 9 (lesion-crop) both
train on only 949 rows. This traces exactly where the other ~1500 rows
are being dropped.

Run from the project root:
    python trace_row_loss.py
"""
import pandas as pd

df = pd.read_csv("data/metadata/bbox_metadata_train.csv")
print(f"Total rows in bbox_metadata_train.csv: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

if "has_bbox" in df.columns:
    valid = df[df["has_bbox"] == True]  # noqa: E712
    print(f"Rows with has_bbox == True: {len(valid)}")
else:
    valid = df
    print("No 'has_bbox' column found -- using all rows.")

# Check for duplicate image_ids (a known CBIS-DDSM quirk: one row per
# FINDING, not per image -- an image with 2 lesions can appear twice).
if "image_id" in valid.columns:
    dupe_count = valid["image_id"].duplicated().sum()
    print(f"Duplicate image_id rows (same image, multiple findings): {dupe_count}")
    print(f"Unique image_ids: {valid['image_id'].nunique()}")

# Check for the specific columns LesionCropDataset/LesionSegmentationDataset
# require to be non-null.
for col in ["image_file_path_resolved", "mask_path_used", "pathology_binary",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
    if col in valid.columns:
        n_null = valid[col].isna().sum()
        print(f"NaN count in '{col}': {n_null}")
    else:
        print(f"Column '{col}' NOT FOUND in this file")

# If pathology_binary needed to be merged in from train_split.csv (as the
# Phase 9/13 console output suggested: "Merged 'pathology_binary' into
# bbox metadata from ...train_split.csv"), check whether that merge could
# be dropping rows due to a key mismatch.
try:
    train_split = pd.read_csv("data/metadata/train_split.csv")
    print(f"\ntrain_split.csv: {len(train_split)} rows, "
          f"{train_split['image_id'].nunique()} unique image_ids")
    common_ids = set(valid["image_id"]) & set(train_split["image_id"])
    print(f"image_ids present in BOTH bbox_metadata_train.csv and train_split.csv: {len(common_ids)}")
    only_in_bbox = set(valid["image_id"]) - set(train_split["image_id"])
    print(f"image_ids in bbox_metadata but NOT in train_split.csv: {len(only_in_bbox)}")
except Exception as e:
    print(f"\nCould not cross-check against train_split.csv: {e}")

# Finally, show exactly what a strict dropna on the required columns leaves.
required_cols = [c for c in ["image_file_path_resolved", "mask_path_used"] if c in valid.columns]
after_dropna = valid.dropna(subset=required_cols)
print(f"\nAfter dropna on {required_cols}: {len(after_dropna)} rows remain")
print("(This is the number LesionSegmentationDataset/LesionCropDataset actually use.)")