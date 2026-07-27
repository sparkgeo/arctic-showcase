from dataclasses import dataclass

from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS, GRID_SIZE, PATCH_SIZE
from training.data_loader.chip import Chip


@dataclass(frozen=True)
class PatchFeatures:
    chip_id: str
    patch_i: int
    patch_j: int
    valid_fraction: float
    hh_mean: float
    hv_mean: float
    hh_std: float
    hv_std: float
    hv_hh_ratio: float
    ia_mean: float
    distance: float
    amsr2: dict[str, float]
    era5: dict[str, float]


def compute_patch_features(chip: Chip) -> list[PatchFeatures]:
    """
    Subdivides a prepared chip into the 32x32 patch grid and computes every
    patch-grain value that is neither a Clay embedding nor a label.

    Operates only on the resampled arrays and pre-substitution valid mask
    carried by the chip.
    """
    valid = chip.valid_mask  # (256, 256) bool, pre-substitution
    hh = chip.sar[0]  # (256, 256)
    hv = chip.sar[1]  # (256, 256)

    features = []

    for pi in range(GRID_SIZE):
        for pj in range(GRID_SIZE):
            r0, r1 = pi * PATCH_SIZE, (pi + 1) * PATCH_SIZE
            c0, c1 = pj * PATCH_SIZE, (pj + 1) * PATCH_SIZE

            valid_patch = valid[r0:r1, c0:c1]  # (8, 8) bool
            hh_patch = hh[r0:r1, c0:c1]
            hv_patch = hv[r0:r1, c0:c1]

            # Valid fraction: count over the pre-substitution boolean mask
            # (64 pixels per 8x8 patch). Computing this post-substitution
            # would count substituted pixels as valid.
            valid_fraction = float(valid_patch.sum() / 64.0)

            # SAR statistics over valid pixels only -- substituted pixels
            # carry the global band mean, not the local signal, and would
            # otherwise contaminate hh_std/hv_std for mixed patches.
            hh_valid = hh_patch[valid_patch]
            hv_valid = hv_patch[valid_patch]

            if len(hh_valid) > 0:
                hh_mean = float(hh_valid.mean())
                hh_std = float(hh_valid.std())
                hv_mean = float(hv_valid.mean())
                hv_std = float(hv_valid.std())
                # Values are in dB, so subtract rather than divide for the ratio.
                hv_hh_ratio = float((hv_valid - hh_valid).mean())
            else:
                hh_mean = hh_std = hv_mean = hv_std = hv_hh_ratio = float("nan")

            # AMSR2/ERA5 are coarse relative to a patch, so average over the whole
            # 8x8 footprint rather than sampling one centroid pixel -- otherwise a
            # patch's feature value is fragile to whatever that single pixel is
            # (interpolation artifact, coastline crossing). distance_to_land is an
            # ordinal index, not a continuous field, so it stays centroid-sampled.
            amsr2_sample = {
                var: float(chip.amsr2[i, r0:r1, c0:c1].mean()) for i, var in enumerate(AMSR2_BANDS)
            }
            era5_sample = {
                var: float(chip.era5[i, r0:r1, c0:c1].mean()) for i, var in enumerate(ERA5_BANDS)
            }
            rc = r0 + PATCH_SIZE // 2
            cc = c0 + PATCH_SIZE // 2
            distance = float(chip.distance_map[rc, cc])

            # Incidence angle is a smooth geometric quantity resampled from the
            # GCP grid, not a substituted/nodata-prone one -- plain mean, no
            # valid-mask filtering.
            ia_mean = float(chip.incidence_angle[r0:r1, c0:c1].mean())

            features.append(
                PatchFeatures(
                    chip_id=chip.chip_id,
                    patch_i=pi,
                    patch_j=pj,
                    valid_fraction=valid_fraction,
                    hh_mean=hh_mean,
                    hv_mean=hv_mean,
                    hh_std=hh_std,
                    hv_std=hv_std,
                    hv_hh_ratio=hv_hh_ratio,
                    ia_mean=ia_mean,
                    distance=distance,
                    amsr2=amsr2_sample,
                    era5=era5_sample,
                )
            )

    return features
