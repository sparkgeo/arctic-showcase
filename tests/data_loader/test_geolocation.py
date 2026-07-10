from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from training.data_loader.geolocation import (
    build_gcp_interpolators,
    get_chip_geo,
    latlon_encoding,
    parse_acquisition_datetime,
    time_encoding,
)


def test_geolocation_helpers_happy_path() -> None:
    gcp_lines = np.array([0.0, 0.0, 10.0, 10.0])
    gcp_samps = np.array([0.0, 10.0, 0.0, 10.0])
    gcp_lats = np.array([60.0, 60.0, 61.0, 61.0])
    gcp_lons = np.array([-80.0, -79.0, -80.0, -79.0])
    gcp_angles = np.array([30.0, 31.0, 32.0, 33.0])

    interpolators = build_gcp_interpolators(gcp_lines, gcp_samps, gcp_lats, gcp_lons, gcp_angles)
    lat, lon, angle = get_chip_geo(interpolators, row_center=5.0, col_center=5.0)
    assert lat == pytest.approx(60.5)
    assert lon == pytest.approx(-79.5)
    assert angle == pytest.approx(31.5)

    scene_path = Path("S1A_EW_GRDM_1SDH_20180124T194759_20180124T194859_020301_022AA4.nc")
    dt = parse_acquisition_datetime(scene_path)
    assert dt == datetime(2018, 1, 24, 19, 47, 59, tzinfo=UTC)

    assert time_encoding(dt).shape == (4,)
    assert latlon_encoding(lat, lon).shape == (4,)
