import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom


def resample_to_sar(
    arr: NDArray[np.float32], target_h: int, target_w: int, order: int = 1
) -> NDArray[np.float32]:
    """Resample arr to (target_h, target_w). order=1 is bilinear, order=0 is nearest-neighbour."""
    zoom_h = target_h / arr.shape[0]
    zoom_w = target_w / arr.shape[1]
    result: NDArray[np.float32] = zoom(arr.astype(np.float32), (zoom_h, zoom_w), order=order)
    return result
