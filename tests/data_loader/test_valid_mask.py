import numpy as np
from training.data_loader.valid_mask import compute_valid_mask, fill_invalid


def test_compute_valid_mask_then_fill_invalid_pixels() -> None:
    sar_primary = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    distance_map = np.array([[5.0, 5.0], [0.0, 5.0]], dtype=np.float32)

    valid_mask = compute_valid_mask(sar_primary, distance_map)
    assert valid_mask.tolist() == [[True, False], [False, True]]

    result = fill_invalid(sar_primary, valid_mask, fill=-1.0)
    assert result.tolist() == [[1.0, -1.0], [-1.0, 3.0]]
    assert result is sar_primary
