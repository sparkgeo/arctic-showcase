import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from training.data_loader.ancillary import resample_ancillary
from training.data_loader.bands import CHIP_SIZE
from training.data_loader.chip import Chip, SceneArrays
from training.data_loader.geolocation import (
    build_gcp_interpolators,
    get_chip_geo,
    latlon_encoding,
    parse_acquisition_datetime,
    time_encoding,
)
from training.data_loader.labels import build_chart_ct
from training.data_loader.scene_reader import read_scene
from training.data_loader.tiling import chip_bounds
from training.data_loader.valid_mask import compute_valid_mask, fill_invalid

logger = logging.getLogger(__name__)


def load_scene(scene_path: Path, band_means: dict[str, float]) -> SceneArrays:
    raw = read_scene(scene_path)

    valid_mask = compute_valid_mask(raw.sar["nersc_sar_primary"], raw.distance_map)
    for var, arr in raw.sar.items():
        fill_invalid(arr, valid_mask, band_means.get(var, 0.0))

    gcp = build_gcp_interpolators(
        raw.gcp_lines, raw.gcp_samps, raw.gcp_lats, raw.gcp_lons, raw.gcp_angles
    )
    ancillary = resample_ancillary(raw, gcp.angles_2d, valid_mask, band_means)
    chart_ct = build_chart_ct(raw.poly_chart, raw.poly_codes)
    acq_dt = parse_acquisition_datetime(scene_path)

    return SceneArrays(
        scene_id=raw.scene_id,
        sar_h=raw.sar_h,
        sar_w=raw.sar_w,
        sar=raw.sar,
        amsr2=ancillary.amsr2,
        era5=ancillary.era5,
        distance_map=ancillary.distance_map,
        incidence_angle=ancillary.incidence_angle,
        valid_mask=valid_mask,
        chart_ct=chart_ct,
        gcp=gcp,
        time_encoding=time_encoding(acq_dt),
    )


def yield_chips(scene: SceneArrays, chip_size: int = CHIP_SIZE) -> Iterator[Chip]:
    n_total = 0
    n_skipped = 0

    for r0, r1, c0, c1 in chip_bounds(scene.sar_h, scene.sar_w, chip_size):
        n_total += 1
        chip_valid = scene.valid_mask[r0:r1, c0:c1]
        if not chip_valid.any():
            n_skipped += 1
            continue

        centroid_lat, centroid_lon, _ = get_chip_geo(
            scene.gcp, r0 + (r1 - r0) / 2.0, c0 + (c1 - c0) / 2.0
        )

        yield Chip(
            sar=np.stack(
                [
                    scene.sar["nersc_sar_primary"][r0:r1, c0:c1],
                    scene.sar["nersc_sar_secondary"][r0:r1, c0:c1],
                ]
            ),
            amsr2=scene.amsr2[:, r0:r1, c0:c1].copy(),
            era5=scene.era5[:, r0:r1, c0:c1].copy(),
            distance_map=scene.distance_map[r0:r1, c0:c1].copy(),
            incidence_angle=scene.incidence_angle[r0:r1, c0:c1].copy(),
            valid_mask=chip_valid.copy(),
            chart_ct=scene.chart_ct[r0:r1, c0:c1].copy(),
            chip_row_start=r0,
            chip_col_start=c0,
            time_encoding=scene.time_encoding,
            latlon_encoding=latlon_encoding(centroid_lat, centroid_lon),
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            scene_id=scene.scene_id,
            chip_id=f"{scene.scene_id}_r{r0:05d}_c{c0:05d}",
        )

    if n_skipped:
        logger.info("%s: skipped %d/%d fully-invalid chips", scene.scene_id, n_skipped, n_total)
