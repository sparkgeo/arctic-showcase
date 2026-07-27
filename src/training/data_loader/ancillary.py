from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS
from training.data_loader.resampling import resample_to_sar
from training.data_loader.scene_reader import RawScene


@dataclass(frozen=True)
class ResampledAncillary:
    amsr2: NDArray[np.float32]  # (14, H, W)
    era5: NDArray[np.float32]  # (6, H, W)
    distance_map: NDArray[np.uint8]  # (H, W)
    incidence_angle: NDArray[np.float32]  # (H, W)


def resample_ancillary(raw: RawScene, angles_2d: NDArray[np.float64]) -> ResampledAncillary:
    """Resamples AMSR2, ERA5, and GCP-grid incidence angle up to SAR resolution.
    distance_map is already native SAR resolution, so it passes through unchanged.

    AMSR2/ERA5 are never fed to Clay and are physically valid over land and open
    water alike (unlike SAR, which has no return over land), so the SAR/land valid
    mask does not apply to them. Values are resampled as-is, with no substitution.
    Genuine sensor/product nodata is already NaN by this point: xarray's default
    mask_and_scale decodes each variable's declared _FillValue on read in
    scene_reader.read_scene, before raw ever reaches this function.
    """
    amsr2 = np.stack(
        [resample_to_sar(raw.amsr2_raw[var], raw.sar_h, raw.sar_w) for var in AMSR2_BANDS]
    )
    era5 = np.stack(
        [resample_to_sar(raw.era5_raw[var], raw.sar_h, raw.sar_w) for var in ERA5_BANDS]
    )
    incidence_angle = resample_to_sar(angles_2d.astype(np.float32), raw.sar_h, raw.sar_w)

    return ResampledAncillary(
        amsr2=amsr2,
        era5=era5,
        distance_map=raw.distance_map.astype(np.uint8),
        incidence_angle=incidence_angle,
    )
