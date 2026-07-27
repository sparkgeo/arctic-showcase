import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom

# Below this fraction of contributing weight, a resampled output pixel is treated
# as having no valid source data at all (see resample_to_sar).
_NAN_WEIGHT_EPS = 1e-6


def resample_to_sar(
    arr: NDArray[np.float32], target_h: int, target_w: int, order: int = 1
) -> NDArray[np.float32]:
    """Resample arr to (target_h, target_w). order=1 is bilinear, order=0 is nearest-neighbour.

    NaN-safe: plain scipy.ndimage.zoom interpolates straight through NaNs, so a single
    genuine gap in arr would smear into a neighbourhood of output pixels wider than the
    gap itself. Instead, zoom the data and a validity mask separately and divide, so each
    output pixel is a weighted average of only its valid source neighbours; only pixels
    with (near) zero valid weight -- fully within the source gap -- come out as NaN.
    """
    zoom_h = target_h / arr.shape[0]
    zoom_w = target_w / arr.shape[1]

    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        result: NDArray[np.float32] = zoom(arr.astype(np.float32), (zoom_h, zoom_w), order=order)
        return result

    filled = np.where(nan_mask, 0.0, arr).astype(np.float32)
    weight = (~nan_mask).astype(np.float32)

    zoomed_vals: NDArray[np.float32] = zoom(filled, (zoom_h, zoom_w), order=order)
    zoomed_weight: NDArray[np.float32] = zoom(weight, (zoom_h, zoom_w), order=order)

    with np.errstate(invalid="ignore", divide="ignore"):
        blended: NDArray[np.float32] = zoomed_vals / zoomed_weight
    blended[zoomed_weight <= _NAN_WEIGHT_EPS] = np.nan
    return blended.astype(np.float32)
