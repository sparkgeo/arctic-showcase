from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS
from training.data_loader.geolocation import GcpInterpolators
from training.data_loader.resampling import resample_to_sar
from training.data_loader.scene_reader import RawScene


@dataclass(frozen=True)
class ResampledAncillary:
    amsr2: NDArray[np.float32]  # (14, H, W)
    era5: NDArray[np.float32]  # (6, H, W)
    distance_map: NDArray[np.uint8]  # (H, W)
    incidence_angle: NDArray[np.float32]  # (H, W)


def resample_ancillary(raw: RawScene, gcp: GcpInterpolators) -> ResampledAncillary:
    """Resamples AMSR2, ERA5, and GCP-grid incidence angle up to SAR resolution.
    distance_map is already native SAR resolution, so it passes through unchanged.

    AMSR2/ERA5 are never fed to Clay and are physically valid over land and open
    water alike (unlike SAR, which has no return over land), so the SAR/land valid
    mask does not apply to them. Values are resampled as-is, with no substitution.
    Genuine sensor/product nodata is already NaN by this point: xarray's default
    mask_and_scale decodes each variable's declared _FillValue on read in
    scene_reader.read_scene, before raw ever reaches this function.

    The incidence angle is evaluated through the GCP position-aware interpolator
    (the same one geolocation.get_chip_geo uses for chip centroids) rather than
    zoom()'d from the raw GCP grid, since zoom assumes the GCPs are evenly spaced
    across the scene, which isn't guaranteed.
    """
    amsr2 = np.stack(
        [resample_to_sar(raw.amsr2_raw[var], raw.sar_h, raw.sar_w) for var in AMSR2_BANDS]
    )
    era5 = np.stack(
        [resample_to_sar(raw.era5_raw[var], raw.sar_h, raw.sar_w) for var in ERA5_BANDS]
    )
    row_idx, col_idx = np.mgrid[0 : raw.sar_h, 0 : raw.sar_w]
    points = np.stack([row_idx.ravel(), col_idx.ravel()], axis=-1)
    incidence_angle = gcp.incidence_angle(points).reshape(raw.sar_h, raw.sar_w).astype(np.float32)

    return ResampledAncillary(
        amsr2=amsr2,
        era5=era5,
        distance_map=raw.distance_map.astype(np.uint8),
        incidence_angle=incidence_angle,
    )
