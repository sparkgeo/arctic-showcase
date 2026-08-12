import numpy as np

from training.data_loader.ancillary import resample_ancillary
from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS
from training.data_loader.geolocation import build_gcp_interpolators
from training.data_loader.resampling import resample_to_sar
from training.data_loader.scene_reader import RawScene

# A 2x2 GCP grid (not a single point) is required to build a real
# RegularGridInterpolator -- mirrors the fixture in test_geolocation.py.
_GCP_LINES = np.array([0.0, 0.0, 10.0, 10.0])
_GCP_SAMPS = np.array([0.0, 10.0, 0.0, 10.0])
_GCP_LATS = np.array([60.0, 60.0, 61.0, 61.0])
_GCP_LONS = np.array([-80.0, -79.0, -80.0, -79.0])
_GCP_ANGLES = np.array([30.0, 30.0, 30.0, 30.0])


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
        gcp_lines=_GCP_LINES,
        gcp_samps=_GCP_SAMPS,
        gcp_lats=_GCP_LATS,
        gcp_lons=_GCP_LONS,
        gcp_angles=_GCP_ANGLES,
    )
    gcp = build_gcp_interpolators(_GCP_LINES, _GCP_SAMPS, _GCP_LATS, _GCP_LONS, _GCP_ANGLES)

    ancillary = resample_ancillary(raw, gcp)

    assert ancillary.amsr2.shape == (len(AMSR2_BANDS), sar_h, sar_w)
    assert ancillary.era5.shape == (len(ERA5_BANDS), sar_h, sar_w)
    assert ancillary.incidence_angle.shape == (sar_h, sar_w)
    assert np.allclose(ancillary.incidence_angle, 30.0)
    assert np.array_equal(ancillary.distance_map, raw.distance_map.astype(np.uint8))


def test_resample_ancillary_does_not_substitute_amsr2_era5() -> None:
    """AMSR2/ERA5 are physically valid over land/SAR-nodata, unlike SAR itself, so
    resample_ancillary must not apply any land/SAR-nodata substitution to them --
    genuine sensor nodata (NaN in the raw arrays) must survive untouched too."""
    sar_h, sar_w = 4, 4
    amsr2_raw = np.full((2, 2), 200.0, dtype=np.float32)
    amsr2_raw[0, 0] = np.nan  # genuine sensor/product nodata
    raw = RawScene(
        scene_id="scene",
        sar_h=sar_h,
        sar_w=sar_w,
        sar={
            "nersc_sar_primary": np.full((sar_h, sar_w), -15.0, dtype=np.float32),
            "nersc_sar_secondary": np.full((sar_h, sar_w), -22.0, dtype=np.float32),
        },
        distance_map=np.full((sar_h, sar_w), 0.0, dtype=np.float32),  # all land
        amsr2_raw={band: amsr2_raw.copy() for band in AMSR2_BANDS},
        era5_raw={band: np.full((2, 2), 250.0, dtype=np.float32) for band in ERA5_BANDS},
        poly_chart=np.zeros((sar_h, sar_w)),
        poly_codes=np.array(["poly_id;CT"]),
        gcp_lines=_GCP_LINES,
        gcp_samps=_GCP_SAMPS,
        gcp_lats=_GCP_LATS,
        gcp_lons=_GCP_LONS,
        gcp_angles=_GCP_ANGLES,
    )
    gcp = build_gcp_interpolators(_GCP_LINES, _GCP_SAMPS, _GCP_LATS, _GCP_LONS, _GCP_ANGLES)

    ancillary = resample_ancillary(raw, gcp)

    # scene is entirely land -- the old bug substituted every pixel here with 0.0
    expected_amsr2 = resample_to_sar(amsr2_raw, sar_h, sar_w)
    expected_era5 = resample_to_sar(raw.era5_raw[ERA5_BANDS[0]], sar_h, sar_w)
    assert np.array_equal(ancillary.amsr2[0], expected_amsr2, equal_nan=True)
    assert np.array_equal(ancillary.era5[0], expected_era5, equal_nan=True)
    assert not np.any(ancillary.era5 == 0.0)
    assert np.isnan(ancillary.amsr2).any()
