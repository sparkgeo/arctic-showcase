from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from training.data_loader.geolocation import GcpInterpolators


@dataclass(frozen=True)
class Chip:
    sar: NDArray[np.float32]
    amsr2: NDArray[np.float32]
    era5: NDArray[np.float32]
    distance_map: NDArray[np.uint8]
    incidence_angle: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    chart_ct: NDArray[np.float32]
    chip_row_start: int
    chip_col_start: int
    time_encoding: NDArray[np.float32]
    latlon_encoding: NDArray[np.float32]
    centroid_lat: float
    centroid_lon: float
    scene_id: str
    chip_id: str


@dataclass(frozen=True)
class SceneArrays:
    scene_id: str
    sar_h: int
    sar_w: int
    sar: dict[str, NDArray[np.float32]]
    amsr2: NDArray[np.float32]
    era5: NDArray[np.float32]
    distance_map: NDArray[np.uint8]
    incidence_angle: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    chart_ct: NDArray[np.float32]
    gcp: GcpInterpolators
    time_encoding: NDArray[np.float32]
