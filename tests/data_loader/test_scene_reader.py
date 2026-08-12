from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr

from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS
from training.data_loader.scene_reader import read_scene


def test_read_scene_loads_every_required_variable(
    tmp_path: Path, make_scene_dataset: Callable[[int, int], xr.Dataset]
) -> None:
    sar_h, sar_w = 4, 4
    scene_path = tmp_path / "S1A_EW_GRDM_1SDH_20180124T194759_test.nc"
    make_scene_dataset(sar_h, sar_w).to_netcdf(scene_path, engine="netcdf4")

    raw = read_scene(scene_path)

    assert raw.scene_id == scene_path.stem
    assert raw.sar_h == sar_h
    assert raw.sar_w == sar_w
    assert raw.sar["nersc_sar_primary"].shape == (sar_h, sar_w)
    assert raw.sar["nersc_sar_primary"].dtype == np.float32
    assert set(raw.amsr2_raw) == set(AMSR2_BANDS)
    assert set(raw.era5_raw) == set(ERA5_BANDS)
    assert raw.gcp_lats.shape == (4,)
