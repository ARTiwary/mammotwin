"""
The CBIS-DDSM case-description CSVs store paths in the ORIGINAL DICOM
export structure, e.g.:

    Mass-Training_P_00001_LEFT_CC/
        1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/   <- study/series UID (A)
            1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/  <- deeper series UID (B)
                000000.dcm

The Kaggle JPEG mirror flattens this: your local `jpeg/` folder contains
one directory per deepest series UID (UID "B" above), e.g.:

    jpeg/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/
        1-263.jpg

The `.dcm` filename is dropped entirely -- the real image is whatever
`.jpg` file(s) sit inside that UID folder. For a full-mammogram image
there's normally exactly one file. For "cropped image" and "ROI mask"
columns, the two can sometimes share the same series UID and folder (a
known CBIS-DDSM quirk), so that folder may contain 2 files -- in that case
we can't fully disambiguate here; n_candidates on the output flags this
so it can be resolved manually when segmentation work (Phase 8) needs it.
"""

import os


def extract_series_uid(raw_path: str):
    """The deepest series UID is the path component immediately before the
    filename, e.g. ".../<UID_A>/<UID_B>/000000.dcm" -> UID_B."""
    if raw_path is None or isinstance(raw_path, float):  # NaN
        return None
    normalized = raw_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    if len(parts) < 2:
        return None
    return parts[-2]


def resolve_kaggle_path(raw_path: str, local_jpeg_root: str):
    """
    Return (resolved_path_or_None, n_candidates_found_in_folder).

    resolved_path is None if the series UID can't be extracted, or its
    folder doesn't exist locally. n_candidates tells you how many image
    files were found in that folder -- 1 is the unambiguous, expected case;
    0 means the folder is missing/empty; 2+ means this series UID is
    shared between two roles (e.g. cropped image + mask) and the first
    file (sorted) was picked arbitrarily.
    """
    uid = extract_series_uid(raw_path)
    if uid is None:
        return None, 0

    folder = os.path.join(local_jpeg_root, uid)
    if not os.path.isdir(folder):
        return None, 0

    files = sorted(f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        return None, 0

    return os.path.join(folder, files[0]), len(files)


def get_all_candidates(raw_path: str, local_jpeg_root: str) -> list:
    """Return ALL image file paths found in the raw_path's series UID folder
    (not just the first, sorted, arbitrary pick that resolve_kaggle_path
    returns). Needed to disambiguate mask-vs-cropped-image when a series UID
    is shared between the two roles (see roi_utils.py)."""
    uid = extract_series_uid(raw_path)
    if uid is None:
        return []
    folder = os.path.join(local_jpeg_root, uid)
    if not os.path.isdir(folder):
        return []
    files = sorted(f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return [os.path.join(folder, f) for f in files]


def add_resolved_paths(df, local_jpeg_root: str):
    """Add `<col>_resolved` and `<col>_n_candidates` for each of the three
    CBIS-DDSM path columns present in df."""
    df = df.copy()
    for col in ["image_file_path", "cropped_image_file_path", "roi_mask_file_path"]:
        if col not in df.columns:
            continue
        resolved, n_candidates = [], []
        for raw in df[col]:
            path, n = resolve_kaggle_path(raw, local_jpeg_root)
            resolved.append(path)
            n_candidates.append(n)
        df[col + "_resolved"] = resolved
        df[col + "_n_candidates"] = n_candidates
    return df


if __name__ == "__main__":
    # Self-test against a folder structure that mimics the user's real setup.
    import tempfile

    raw_path = ("Mass-Training_P_00001_LEFT_CC/"
                "1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/"
                "1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/"
                "000000.dcm")

    with tempfile.TemporaryDirectory() as tmp_root:
        uid = "1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515"
        series_folder = os.path.join(tmp_root, uid)
        os.makedirs(series_folder)
        with open(os.path.join(series_folder, "1-263.jpg"), "wb") as f:
            f.write(b"fake jpg bytes")

        resolved, n = resolve_kaggle_path(raw_path, tmp_root)
        print("Extracted UID:", extract_series_uid(raw_path))
        print("Resolved path:", resolved)
        print("n_candidates: ", n)
        assert resolved is not None and n == 1, "Self-test FAILED"
        print("Self-test PASSED")