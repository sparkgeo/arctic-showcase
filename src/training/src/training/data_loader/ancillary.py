from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS
from training.data_loader.resampling import resample_to_sar
from training.data_loader.scene_reader import RawScene
from training.data_loader.valid_mask import fill_invalid

@dataclass(frozen=True)
class ResampledAncillary:
    amsr2: NDArray[np.float32]  # (14, H, W)
    era5: NDArray[np.float32]  # (6, H, W)
    distance_map: NDArray[np.uint8]  # (H, W)
    incidence_angle: NDArray[np.float32]  # (H, W)


def _resample_and_fill(
    raw: NDArray[np.float32],
    sar_h: int,
    sar_w: int,
    valid_mask: NDArray[np.bool_],
    fill: float,
) -> NDArray[np.float32]:
    return fill_invalid(resample_to_sar(raw, sar_h, sar_w), valid_mask, fill)


def resample_ancillary(
    raw: RawScene,
    angles_2d: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    band_means: dict[str, float],
) -> ResampledAncillary:
    """Resamples AMSR2, ERA5, and GCP-grid incidence angle up to SAR resolution.
    distance_map is already native SAR resolution, so it passes through unchanged."""
    amsr2 = np.stack(
        [
            _resample_and_fill(
                raw.amsr2_raw[var], raw.sar_h, raw.sar_w, valid_mask, band_means.get(var, 0.0)
            )
            for var in AMSR2_BANDS
        ]
    )
    era5 = np.stack(
        [
            _resample_and_fill(
                raw.era5_raw[var], raw.sar_h, raw.sar_w, valid_mask, band_means.get(var, 0.0)
            )
            for var in ERA5_BANDS
        ]
    )
    incidence_angle = resample_to_sar(angles_2d.astype(np.float32), raw.sar_h, raw.sar_w)

    return ResampledAncillary(
        amsr2=amsr2,
        era5=era5,
        distance_map=raw.distance_map.astype(np.uint8),
        incidence_angle=incidence_angle,
    )
