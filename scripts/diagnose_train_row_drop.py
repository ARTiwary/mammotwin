import pandas as pd

df = pd.read_csv("data/metadata/bbox_metadata_train.csv")
print("Total rows:", len(df))
print("has_bbox counts:", df["has_bbox"].value_counts(dropna=False).to_dict())

valid = df[df["has_bbox"] == True]
print("has_bbox==True rows:", len(valid))
print("image_file_path_resolved NaN count (within valid):", valid["image_file_path_resolved"].isna().sum())
print("mask_path_used NaN count (within valid):", valid["mask_path_used"].isna().sum())

print()
print("Sample of valid rows with a NaN in either column:")
bad = valid[valid["image_file_path_resolved"].isna() | valid["mask_path_used"].isna()]
print(bad[["image_id", "has_bbox", "image_file_path_resolved", "mask_path_used"]].head(10))

print()
print("Duplicate image_id count (multiple lesions per image?):", valid["image_id"].duplicated().sum())
