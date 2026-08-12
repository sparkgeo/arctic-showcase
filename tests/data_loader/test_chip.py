import numpy as np
from scipy.interpolate import RegularGridInterpolator

from training.data_loader.chip import Chip, SceneArrays
from training.data_loader.geolocation import GcpInterpolators


def test_chip_and_scene_arrays_store_their_fields() -> None:
    sar = np.zeros((2, 4, 4), dtype=np.float32)
    amsr2 = np.zeros((14, 4, 4), dtype=np.float32)
    era5 = np.zeros((6, 4, 4), dtype=np.float32)
    distance_map = np.zeros((4, 4), dtype=np.uint8)
    incidence_angle = np.zeros((4, 4), dtype=np.float32)
    valid_mask = np.ones((4, 4), dtype=np.bool_)
    chart_ct = np.zeros((4, 4), dtype=np.float32)
    time_enc = np.zeros(4, dtype=np.float32)
    latlon_enc = np.zeros(4, dtype=np.float32)

    chip = Chip(
        sar=sar,
        amsr2=amsr2,
        era5=era5,
        distance_map=distance_map,
        incidence_angle=incidence_angle,
        valid_mask=valid_mask,
        chart_ct=chart_ct,
        chip_row_start=0,
        chip_col_start=0,
        time_encoding=time_enc,
        latlon_encoding=latlon_enc,
        centroid_lat=60.0,
        centroid_lon=-80.0,
        scene_id="scene",
        chip_id="scene_r00000_c00000",
    )
    assert chip.scene_id == "scene"
    assert chip.sar.shape == (2, 4, 4)

    axis = np.array([0.0, 1.0])
    grid = np.array([[0.0, 1.0], [2.0, 3.0]])
    interp = RegularGridInterpolator((axis, axis), grid)
    gcp = GcpInterpolators(lat=interp, lon=interp, incidence_angle=interp, angles_2d=grid)

    scene = SceneArrays(
        scene_id="scene",
        sar_h=4,
        sar_w=4,
        sar={"nersc_sar_primary": sar[0], "nersc_sar_secondary": sar[1]},
        amsr2=amsr2,
        era5=era5,
        distance_map=distance_map,
        incidence_angle=incidence_angle,
        valid_mask=valid_mask,
        chart_ct=chart_ct,
        gcp=gcp,
        time_encoding=time_enc,
    )
    assert scene.sar_h == 4
    assert scene.gcp is gcp
