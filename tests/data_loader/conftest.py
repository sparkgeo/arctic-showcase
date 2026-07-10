from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS


@pytest.fixture
def make_scene_dataset() -> Callable[[int, int], xr.Dataset]:
    """Builds a minimal in-memory Dataset containing every variable read_scene expects."""

    def _make(sar_h: int, sar_w: int) -> xr.Dataset:
        data_vars: dict[str, object] = {
            "nersc_sar_primary": (("y", "x"), np.full((sar_h, sar_w), -15.0, dtype=np.float32)),
            "nersc_sar_secondary": (("y", "x"), np.full((sar_h, sar_w), -22.0, dtype=np.float32)),
            "distance_map": (("y", "x"), np.full((sar_h, sar_w), 5.0, dtype=np.float32)),
            "polygon_icechart": (("y", "x"), np.ones((sar_h, sar_w), dtype=np.float64)),
            "polygon_codes": (("poly_row",), np.array(["poly_id;CT", "1;80"])),
        }
        for band in AMSR2_BANDS:
            data_vars[band] = (("amsr_y", "amsr_x"), np.full((2, 2), 200.0, dtype=np.float32))
        for band in ERA5_BANDS:
            data_vars[band] = (("era5_y", "era5_x"), np.full((2, 2), 250.0, dtype=np.float32))

        coords = {
            "sar_grid_line": (("gcp",), np.array([0.0, 0.0, sar_h - 1.0, sar_h - 1.0])),
            "sar_grid_sample": (("gcp",), np.array([0.0, sar_w - 1.0, 0.0, sar_w - 1.0])),
            "sar_grid_latitude": (("gcp",), np.array([60.0, 60.0, 61.0, 61.0])),
            "sar_grid_longitude": (("gcp",), np.array([-80.0, -79.0, -80.0, -79.0])),
            "sar_grid_incidenceangle": (("gcp",), np.array([30.0, 31.0, 32.0, 33.0])),
        }
        return xr.Dataset(data_vars=data_vars, coords=coords)

    return _make
