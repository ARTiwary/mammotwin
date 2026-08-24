"""
Phase 3 diagnostic: run this once to figure out exactly why paths aren't
resolving, then delete it — it's not part of the permanent pipeline.

Usage:
    python scripts/diagnose_paths.py --mass-train "C:\\...\\mass_case_description_train_set.csv" --jpeg-dir "C:\\...\\Datasets\\jpeg"
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mass-train", required=True)
    parser.add_argument("--jpeg-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.mass_train)

    print("=== Columns in the CSV ===")
    print(list(df.columns))

    path_col = "image file path" if "image file path" in df.columns else None
    if path_col is None:
        print("\nCould not find 'image file path' column — printing first row instead:")
        print(df.iloc[0])
        return

    print(f"\n=== First 3 raw values of '{path_col}' ===")
    for v in df[path_col].head(3):
        print(repr(v))

    print(f"\n=== Contents of your --jpeg-dir ({args.jpeg_dir}) ===")
    if not os.path.exists(args.jpeg_dir):
        print("  THIS PATH DOES NOT EXIST. Double check --jpeg-dir.")
        return
    entries = os.listdir(args.jpeg_dir)
    print(f"  {len(entries)} entries found. First 5:")
    for e in entries[:5]:
        full = os.path.join(args.jpeg_dir, e)
        kind = "DIR" if os.path.isdir(full) else "FILE"
        print(f"    [{kind}] {e}")

    # If the first entry is a directory, peek one level deeper too.
    if entries and os.path.isdir(os.path.join(args.jpeg_dir, entries[0])):
        deeper = os.listdir(os.path.join(args.jpeg_dir, entries[0]))
        print(f"\n  Contents of {entries[0]} (first 5):")
        for e in deeper[:5]:
            print(f"    {e}")


if __name__ == "__main__":
    main()