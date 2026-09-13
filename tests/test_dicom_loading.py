"""
Phase 16: "Test DICOM loading and image orientation" -- specifically the
MONOCHROME1/MONOCHROME2 inversion handling in src/data/image_io.py, which
is the single easiest DICOM bug to introduce silently (getting this wrong
doesn't crash anything, it just quietly inverts every downstream analysis).

Synthetic, minimal-but-valid DICOM files are generated with pydicom so
these tests don't depend on any real patient data being present.
"""
import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom", reason="pydicom not installed")

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.data.image_io import load_dicom, load_image, is_dicom


def _make_synthetic_dicom(path, pixel_array: np.ndarray, photometric: str = "MONOCHROME2",
                           rescale_slope: float = 1.0, rescale_intercept: float = 0.0):
    """Writes a minimal, valid DICOM file to `path` with the given pixel
    data and photometric interpretation -- just enough tags for pydicom to
    round-trip it, matching what load_dicom() actually reads."""
    pixel_array = pixel_array.astype(np.uint16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.Rows, ds.Columns = pixel_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept
    ds.PixelData = pixel_array.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.save_as(str(path), write_like_original=False)


def test_is_dicom_extension_detection():
    assert is_dicom("scan.dcm") is True
    assert is_dicom("SCAN.DICOM") is True
    assert is_dicom("scan.jpg") is False
    assert is_dicom("scan.png") is False


def test_load_dicom_monochrome2_is_not_inverted(tmp_path):
    """MONOCHROME2 is the 'normal' convention (higher value = brighter) --
    load_dicom must NOT flip it."""
    path = tmp_path / "mono2.dcm"
    pixel_array = np.zeros((16, 16), dtype=np.uint16)
    pixel_array[0, 0] = 4000   # a bright pixel
    pixel_array[5, 5] = 100    # a dark pixel
    _make_synthetic_dicom(path, pixel_array, photometric="MONOCHROME2")

    img = load_dicom(str(path))
    assert img[0, 0] > img[5, 5], "MONOCHROME2 should not be inverted: bright stays bright"


def test_load_dicom_monochrome1_is_inverted(tmp_path):
    """MONOCHROME1 stores pixel values inverted (0 = bright). load_dicom
    must flip these so downstream code sees a consistent convention
    regardless of which photometric interpretation the file used."""
    path = tmp_path / "mono1.dcm"
    pixel_array = np.zeros((16, 16), dtype=np.uint16)
    pixel_array[0, 0] = 100    # LOW stored value = bright, in MONOCHROME1
    pixel_array[5, 5] = 4000   # HIGH stored value = dark, in MONOCHROME1
    _make_synthetic_dicom(path, pixel_array, photometric="MONOCHROME1")

    img = load_dicom(str(path))
    # After correction, pixel [0,0] (low stored value, "bright" in MONOCHROME1)
    # must end up BRIGHTER than [5,5] in the corrected output.
    assert img[0, 0] > img[5, 5], "MONOCHROME1 pixels must be inverted to a consistent convention"


def test_load_dicom_monochrome1_and_monochrome2_agree_after_correction(tmp_path):
    """The whole point of the inversion fix: two files encoding the SAME
    visual image, one in each photometric convention, must produce the
    same corrected result (up to the max-value reflection used to invert)."""
    visual = np.array([[100, 4000], [2000, 500]], dtype=np.uint16)  # "true" brightness
    mono2_path = tmp_path / "as_mono2.dcm"
    mono1_path = tmp_path / "as_mono1.dcm"

    _make_synthetic_dicom(mono2_path, visual, photometric="MONOCHROME2")
    # Encode the SAME visual image as MONOCHROME1 by pre-inverting the stored values.
    inverted_stored = visual.max() - visual
    _make_synthetic_dicom(mono1_path, inverted_stored, photometric="MONOCHROME1")

    img_from_mono2 = load_dicom(str(mono2_path))
    img_from_mono1 = load_dicom(str(mono1_path))

    # Both should rank pixels in the same brightness order after correction.
    assert np.argsort(img_from_mono2.ravel()).tolist() == np.argsort(img_from_mono1.ravel()).tolist()


def test_load_dicom_applies_rescale_slope_and_intercept(tmp_path):
    path = tmp_path / "rescaled.dcm"
    stored = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    _make_synthetic_dicom(path, stored, rescale_slope=2.0, rescale_intercept=5.0)

    img = load_dicom(str(path))
    expected = stored.astype(np.float32) * 2.0 + 5.0
    np.testing.assert_allclose(img, expected)


def test_load_dicom_output_is_float32(tmp_path):
    path = tmp_path / "dtype_check.dcm"
    _make_synthetic_dicom(path, np.full((8, 8), 500, dtype=np.uint16))
    img = load_dicom(str(path))
    assert img.dtype == np.float32


def test_load_image_dispatches_to_dicom_loader(tmp_path):
    """load_image() should auto-detect a .dcm extension and route to
    load_dicom() rather than the standard-image loader."""
    path = tmp_path / "auto_detect.dcm"
    _make_synthetic_dicom(path, np.full((8, 8), 300, dtype=np.uint16))
    img = load_image(str(path))
    assert img.shape == (8, 8)
    assert img.dtype == np.float32


def test_load_image_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_image("/no/such/file/anywhere.dcm")
