import numpy as np
from training.data_loader.ancillary import resample_ancillary
from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS
from training.data_loader.scene_reader import RawScene


def test_resample_ancillary_upsamples_and_passes_through_the_distance_map() -> None:
    sar_h, sar_w = 4, 4
    raw = RawScene(
        scene_id="scene",
        sar_h=sar_h,
        sar_w=sar_w,
        sar={
            "nersc_sar_primary": np.full((sar_h, sar_w), -15.0, dtype=np.float32),
            "nersc_sar_secondary": np.full((sar_h, sar_w), -22.0, dtype=np.float32),
        },
        distance_map=np.full((sar_h, sar_w), 5.0, dtype=np.float32),
        amsr2_raw={band: np.full((2, 2), 200.0, dtype=np.float32) for band in AMSR2_BANDS},
        era5_raw={band: np.full((2, 2), 250.0, dtype=np.float32) for band in ERA5_BANDS},
        poly_chart=np.zeros((sar_h, sar_w)),
        poly_codes=np.array(["poly_id;CT"]),
        gcp_lines=np.array([0.0]),
        gcp_samps=np.array([0.0]),
        gcp_lats=np.array([60.0]),
        gcp_lons=np.array([-80.0]),
        gcp_angles=np.array([30.0]),
    )
    angles_2d = np.full((2, 2), 30.0, dtype=np.float64)
    valid_mask = np.ones((sar_h, sar_w), dtype=np.bool_)

    ancillary = resample_ancillary(raw, angles_2d, valid_mask, band_means={})

    assert ancillary.amsr2.shape == (len(AMSR2_BANDS), sar_h, sar_w)
    assert ancillary.era5.shape == (len(ERA5_BANDS), sar_h, sar_w)
    assert ancillary.incidence_angle.shape == (sar_h, sar_w)
    assert np.array_equal(ancillary.distance_map, raw.distance_map.astype(np.uint8))
