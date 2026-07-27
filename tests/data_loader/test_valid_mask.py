import numpy as np
from training.data_loader.valid_mask import compute_valid_mask, fill_invalid


def test_compute_valid_mask_then_fill_invalid_pixels() -> None:
    sar_primary = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    sar_secondary = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    distance_map = np.array([[5.0, 5.0], [0.0, 5.0]], dtype=np.float32)

    valid_mask = compute_valid_mask(sar_primary, sar_secondary, distance_map)
    assert valid_mask.tolist() == [[True, False], [False, True]]

    result = fill_invalid(sar_primary, valid_mask, fill=-1.0)
    assert result.tolist() == [[1.0, -1.0], [-1.0, 3.0]]
    assert result is sar_primary


def test_compute_valid_mask_unions_hh_and_hv_nodata() -> None:
    """A pixel valid in one SAR band but NaN in the other (e.g. HV masked near the
    noise floor at far range while HH is still a real return) must be marked
    invalid overall, so fill_invalid substitutes it in both bands rather than
    letting the NaN survive into the band where it wasn't originally present."""
    sar_primary = np.array([[1.0, 2.0, np.nan]], dtype=np.float32)
    sar_secondary = np.array([[np.nan, 2.0, 3.0]], dtype=np.float32)
    distance_map = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)

    valid_mask = compute_valid_mask(sar_primary, sar_secondary, distance_map)
    assert valid_mask.tolist() == [[False, True, False]]

    fill_invalid(sar_secondary, valid_mask, fill=-1.0)
    assert sar_secondary.tolist() == [[-1.0, 2.0, -1.0]]
