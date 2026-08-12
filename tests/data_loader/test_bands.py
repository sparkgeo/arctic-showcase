from training.data_loader.bands import (
    ALL_BANDS,
    AMSR2_BANDS,
    ANCILLARY_BANDS,
    CHIP_SIZE,
    ERA5_BANDS,
    GRID_SIZE,
    N_PATCHES,
    PATCH_SIZE,
    SAR_BANDS,
)


def test_band_constants_are_internally_consistent() -> None:
    assert N_PATCHES == (CHIP_SIZE // PATCH_SIZE) ** 2
    assert GRID_SIZE == CHIP_SIZE // PATCH_SIZE
    assert SAR_BANDS == ["nersc_sar_primary", "nersc_sar_secondary"]
    assert len(AMSR2_BANDS) == 14
    assert len(ERA5_BANDS) == 6
    assert ANCILLARY_BANDS == ["sar_grid_incidenceangle", "distance_map"]
    assert ALL_BANDS == SAR_BANDS + AMSR2_BANDS + ERA5_BANDS
