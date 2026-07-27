import numpy as np
from numpy.typing import NDArray


def compute_valid_mask(
    sar_primary: NDArray[np.float32],
    sar_secondary: NDArray[np.float32],
    distance_map: NDArray[np.float32],
) -> NDArray[np.bool_]:
    """True = valid pixel. Computed before substitution: land is distance_map code 0,
    nodata is NaN in either SAR band -- HH and HV don't necessarily share the same
    nodata footprint (e.g. far-range near-noise-floor masking), so a pixel valid in
    one band but NaN in the other must still be treated as invalid in both."""
    is_land = distance_map == 0
    is_nodata = np.isnan(sar_primary) | np.isnan(sar_secondary)
    valid_mask: NDArray[np.bool_] = ~is_land & ~is_nodata
    return valid_mask


def fill_invalid(
    arr: NDArray[np.float32], valid_mask: NDArray[np.bool_], fill: float
) -> NDArray[np.float32]:
    """In-place substitution. After this, invalid pixels are exactly zero in Clay's
    normalised space (mean - mean = 0 after z-score normalisation)."""
    arr[~valid_mask] = fill
    return arr
