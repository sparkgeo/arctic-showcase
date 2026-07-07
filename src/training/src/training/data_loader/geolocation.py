import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

_S1_DATETIME_RE = re.compile(r"(\d{8}T\d{6})")


@dataclass(frozen=True)
class GcpInterpolators:
    lat: RegularGridInterpolator
    lon: RegularGridInterpolator
    incidence_angle: RegularGridInterpolator
    angles_2d: NDArray[np.float64]


def build_gcp_interpolators(
    gcp_lines: NDArray[np.float64],
    gcp_samps: NDArray[np.float64],
    gcp_lats: NDArray[np.float64],
    gcp_lons: NDArray[np.float64],
    gcp_angles: NDArray[np.float64],
) -> GcpInterpolators:
    # GCP count read dynamically from the array itself (which mirrors sar_grid_points).
    gcp_side = int(np.sqrt(gcp_lines.size))

    # NetCDF storage order is not guaranteed to be row-major.
    sort_idx = np.lexsort((gcp_samps, gcp_lines))
    lines_2d = gcp_lines[sort_idx].reshape(gcp_side, gcp_side)
    samps_2d = gcp_samps[sort_idx].reshape(gcp_side, gcp_side)
    lats_2d = gcp_lats[sort_idx].reshape(gcp_side, gcp_side)
    lons_2d = gcp_lons[sort_idx].reshape(gcp_side, gcp_side)
    angles_2d = gcp_angles[sort_idx].reshape(gcp_side, gcp_side)

    row_axis = lines_2d[:, 0]
    col_axis = samps_2d[0, :]

    interp_lat = RegularGridInterpolator(
        (row_axis, col_axis), lats_2d, method="linear", bounds_error=False, fill_value=None
    )
    interp_lon = RegularGridInterpolator(
        (row_axis, col_axis), lons_2d, method="linear", bounds_error=False, fill_value=None
    )
    interp_angle = RegularGridInterpolator(
        (row_axis, col_axis), angles_2d, method="linear", bounds_error=False, fill_value=None
    )

    return GcpInterpolators(
        lat=interp_lat, lon=interp_lon, incidence_angle=interp_angle, angles_2d=angles_2d
    )


def get_chip_geo(
    interpolators: GcpInterpolators, row_center: float, col_center: float
) -> tuple[float, float, float]:
    """Return (lat, lon, incidence_angle) for a pixel coordinate."""
    pt = np.array([[float(row_center), float(col_center)]])
    lat = float(interpolators.lat(pt)[0])
    lon = float(interpolators.lon(pt)[0])
    angle = float(interpolators.incidence_angle(pt)[0])
    return lat, lon, angle


def parse_acquisition_datetime(scene_path: Path) -> datetime:
    match = _S1_DATETIME_RE.search(scene_path.name)
    if match is None:
        raise ValueError(f"No acquisition timestamp found in {scene_path.name}")
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def time_encoding(dt: datetime) -> NDArray[np.float32]:
    """Clay temporal encoding: [sin/cos week, sin/cos hour]."""
    week = float(dt.isocalendar().week)
    hour = dt.hour + dt.minute / 60.0
    return np.array(
        [
            math.sin(week * 2 * math.pi / 52),
            math.cos(week * 2 * math.pi / 52),
            math.sin(hour * 2 * math.pi / 24),
            math.cos(hour * 2 * math.pi / 24),
        ],
        dtype=np.float32,
    )


def latlon_encoding(lat: float, lon: float) -> NDArray[np.float32]:
    """Clay spatial encoding: [sin/cos lat, sin/cos lon]."""
    return np.array(
        [
            math.sin(lat * math.pi / 180),
            math.cos(lat * math.pi / 180),
            math.sin(lon * math.pi / 180),
            math.cos(lon * math.pi / 180),
        ],
        dtype=np.float32,
    )
