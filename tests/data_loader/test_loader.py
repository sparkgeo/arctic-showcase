from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr

from training.data_loader.bands import AMSR2_BANDS
from training.data_loader.loader import load_scene, yield_chips


def test_load_scene_and_yield_chips_happy_path(
    tmp_path: Path, make_scene_dataset: Callable[[int, int], xr.Dataset]
) -> None:
    sar_h, sar_w = 8, 8
    scene_path = tmp_path / "S1A_EW_GRDM_1SDH_20180124T194759_test.nc"
    make_scene_dataset(sar_h, sar_w).to_netcdf(scene_path, engine="netcdf4")

    scene = load_scene(scene_path, band_means={})
    chips = list(yield_chips(scene, chip_size=4))

    assert len(chips) == 4  # 8x8 scene tiled into 4x4 chips, none fully invalid
    chip = chips[0]
    assert chip.sar.shape == (2, 4, 4)
    assert chip.amsr2.shape == (len(AMSR2_BANDS), 4, 4)
    assert chip.scene_id == scene.scene_id
    assert chip.chip_id == f"{scene.scene_id}_r00000_c00000"


def test_yield_chips_skips_chips_with_no_label_coverage(
    tmp_path: Path, make_scene_dataset: Callable[[int, int], xr.Dataset]
) -> None:
    sar_h, sar_w = 8, 8
    dataset = make_scene_dataset(sar_h, sar_w)

    poly_icechart = dataset["polygon_icechart"].values.copy()
    poly_icechart[:4, :4] = 2.0  # top-left chip belongs to a polygon with no CT value
    dataset["polygon_icechart"] = (("y", "x"), poly_icechart)
    dataset["polygon_codes"] = (("poly_row",), np.array(["poly_id;CT", "1;80", "2;-9"]))

    scene_path = tmp_path / "S1A_EW_GRDM_1SDH_20180124T194759_test.nc"
    dataset.to_netcdf(scene_path, engine="netcdf4")

    scene = load_scene(scene_path, band_means={})
    chips = list(yield_chips(scene, chip_size=4))

    assert len(chips) == 3  # top-left chip has SAR coverage but zero label coverage
    chip_ids = {chip.chip_id for chip in chips}
    assert f"{scene.scene_id}_r00000_c00000" not in chip_ids
