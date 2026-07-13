import numpy as np
from training.data_loader.bands import CHIP_SIZE, GRID_SIZE, N_PATCHES
from training.data_loader.chip import Chip
from training.label_prep import compute_patch_labels


def _make_chip(chart_ct: np.ndarray) -> Chip:
    dummy_2d = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    return Chip(
        sar=np.zeros((2, CHIP_SIZE, CHIP_SIZE), dtype=np.float32),
        amsr2=np.zeros((14, CHIP_SIZE, CHIP_SIZE), dtype=np.float32),
        era5=np.zeros((6, CHIP_SIZE, CHIP_SIZE), dtype=np.float32),
        distance_map=dummy_2d.astype(np.uint8),
        incidence_angle=dummy_2d,
        valid_mask=np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.bool_),
        chart_ct=chart_ct,
        chip_row_start=0,
        chip_col_start=0,
        time_encoding=np.zeros(4, dtype=np.float32),
        latlon_encoding=np.zeros(4, dtype=np.float32),
        centroid_lat=60.0,
        centroid_lon=-80.0,
        scene_id="scene",
        chip_id="scene_r00000_c00000",
    )


def test_compute_patch_labels_happy_path() -> None:
    chart_ct = np.full((CHIP_SIZE, CHIP_SIZE), np.nan, dtype=np.float32)
    chart_ct[0:8, 0:8] = 8.0  # patch (0, 0): pure class 8
    chart_ct[0:8, 8:16] = np.concatenate(  # patch (0, 1): mixed classes 8 and 9
        [np.full((4, 8), 8.0), np.full((4, 8), 9.0)]
    )

    labels = compute_patch_labels(_make_chip(chart_ct))

    assert len(labels) == N_PATCHES == GRID_SIZE * GRID_SIZE

    pure = labels[0]
    assert (pure.patch_i, pure.patch_j) == (0, 0)
    assert pure.valid_class_fraction == 1.0
    assert pure.is_pure is True
    assert pure.label == 8.0
    assert pure.frac_sic[8] == 1.0

    mixed = labels[1]
    assert (mixed.patch_i, mixed.patch_j) == (0, 1)
    assert mixed.is_pure is False
    assert mixed.frac_sic[8] == 0.5
    assert mixed.frac_sic[9] == 0.5
    assert mixed.label == 8.0  # 8*0.5 + 9*0.5 = 8.5, banker's rounding -> 8

    empty = labels[-1]  # bottom-right patch: no chart coverage at all
    assert empty.valid_class_fraction == 0.0
    assert np.isnan(empty.label)
    assert empty.is_pure is False
