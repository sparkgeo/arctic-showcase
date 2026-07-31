import math

import numpy as np
import pytest
from numpy.typing import NDArray

from training.data_loader.bands import AMSR2_BANDS, CHIP_SIZE, ERA5_BANDS, N_PATCHES
from training.data_loader.chip import Chip
from training.patch_features import compute_patch_features


def _make_chip(
    *,
    hh: NDArray[np.float32] | None = None,
    hv: NDArray[np.float32] | None = None,
    valid_mask: NDArray[np.bool_] | None = None,
    amsr2: NDArray[np.float32] | None = None,
    era5: NDArray[np.float32] | None = None,
    distance_map: NDArray[np.float32] | None = None,
    incidence_angle: NDArray[np.float32] | None = None,
) -> Chip:
    zeros_2d = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    return Chip(
        sar=np.stack(
            [
                hh if hh is not None else zeros_2d.copy(),
                hv if hv is not None else zeros_2d.copy(),
            ]
        ),
        amsr2=amsr2
        if amsr2 is not None
        else np.zeros((14, CHIP_SIZE, CHIP_SIZE), dtype=np.float32),
        era5=era5 if era5 is not None else np.zeros((6, CHIP_SIZE, CHIP_SIZE), dtype=np.float32),
        distance_map=(distance_map if distance_map is not None else zeros_2d).astype(np.uint8),
        incidence_angle=incidence_angle if incidence_angle is not None else zeros_2d.copy(),
        valid_mask=valid_mask
        if valid_mask is not None
        else np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.bool_),
        chart_ct=zeros_2d.copy(),
        chip_row_start=0,
        chip_col_start=0,
        time_encoding=np.zeros(4, dtype=np.float32),
        latlon_encoding=np.zeros(4, dtype=np.float32),
        centroid_lat=60.0,
        centroid_lon=-80.0,
        scene_id="scene",
        chip_id="scene_r00000_c00000",
    )


def test_compute_patch_features_covers_the_full_grid() -> None:
    features = compute_patch_features(_make_chip())
    assert len(features) == N_PATCHES


def test_valid_fraction_and_sar_stats_on_a_fully_valid_patch() -> None:
    hh = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    hh[0:4, 0:8] = 10.0
    hh[4:8, 0:8] = 20.0
    hv = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    hv[0:4, 0:8] = -20.0
    hv[4:8, 0:8] = -10.0

    patch = compute_patch_features(_make_chip(hh=hh, hv=hv))[0]  # patch (0, 0)

    assert patch.valid_fraction == 1.0
    assert patch.hh_mean == 15.0
    assert patch.hh_std == 5.0
    assert patch.hv_mean == -15.0
    assert patch.hv_std == 5.0
    assert patch.hv_hh_ratio == -30.0  # dB difference, not a ratio: mean(hv - hh)


def test_partial_validity_excludes_substituted_pixels_from_sar_stats() -> None:
    hh = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    hh[0:4, 8:16] = 999.0  # substituted band-mean stand-in -- must not leak into stats
    hh[4:8, 8:16] = 100.0
    hv = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    hv[0:4, 8:16] = 999.0
    hv[4:8, 8:16] = 50.0
    valid_mask = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.bool_)
    valid_mask[0:4, 8:16] = False

    patch = compute_patch_features(_make_chip(hh=hh, hv=hv, valid_mask=valid_mask))[1]  # (0, 1)

    assert patch.valid_fraction == 0.5
    assert patch.hh_mean == 100.0
    assert patch.hh_std == 0.0
    assert patch.hv_mean == 50.0
    assert patch.hv_hh_ratio == -50.0


def test_fully_invalid_patch_yields_zero_valid_fraction_and_nan_stats() -> None:
    valid_mask = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.bool_)
    valid_mask[248:256, 248:256] = False  # bottom-right patch (31, 31)

    patch = compute_patch_features(_make_chip(valid_mask=valid_mask))[-1]

    assert patch.valid_fraction == 0.0
    assert math.isnan(patch.hh_mean)
    assert math.isnan(patch.hh_std)
    assert math.isnan(patch.hv_mean)
    assert math.isnan(patch.hv_std)
    assert math.isnan(patch.hv_hh_ratio)


def test_ia_mean_is_computed_regardless_of_valid_mask() -> None:
    incidence_angle = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    incidence_angle[0:4, 0:8] = 30.0
    incidence_angle[4:8, 0:8] = 34.0
    valid_mask = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.bool_)  # whole patch marked invalid

    patch = compute_patch_features(
        _make_chip(incidence_angle=incidence_angle, valid_mask=valid_mask)
    )[0]

    assert patch.valid_fraction == 0.0
    assert patch.ia_mean == 32.0  # unaffected by the all-invalid mask


def test_amsr2_and_era5_channel_ordering_and_patch_mean_aggregation() -> None:
    amsr2 = np.zeros((14, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    for i in range(14):
        amsr2[i, 0:8, 0:8] = 200.0 + i
    amsr2[0, 0:4, 0:8] = 190.0  # channel 0: split so the result proves averaging, not a
    amsr2[0, 4:8, 0:8] = 210.0  # single-pixel read (mean is still 200.0)
    era5 = np.zeros((6, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    for i in range(6):
        era5[i, 0:8, 0:8] = 300.0 + i

    patch = compute_patch_features(_make_chip(amsr2=amsr2, era5=era5))[0]

    for i, var in enumerate(AMSR2_BANDS):
        assert patch.amsr2[var] == pytest.approx(200.0 + i)
    for i, var in enumerate(ERA5_BANDS):
        assert patch.era5[var] == pytest.approx(300.0 + i)


def test_distance_is_sampled_at_the_patch_centroid_not_averaged() -> None:
    distance_map = np.full((CHIP_SIZE, CHIP_SIZE), 3.0, dtype=np.float32)
    distance_map[4, 4] = 7.0  # centroid pixel of patch (0, 0): row 0+4, col 0+4

    patch = compute_patch_features(_make_chip(distance_map=distance_map))[0]

    assert patch.distance == 7.0  # a patch mean would read ~3.06, not 7.0
