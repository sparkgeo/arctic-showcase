from dataclasses import dataclass
from pathlib import Path

import boto3
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS, SAR_BANDS


@dataclass(frozen=True)
class RawScene:
    scene_id: str
    sar_h: int
    sar_w: int
    sar: dict[str, NDArray[np.float32]]
    distance_map: NDArray[np.float32]
    amsr2_raw: dict[str, NDArray[np.float32]]
    era5_raw: dict[str, NDArray[np.float32]]
    poly_chart: NDArray[np.generic]
    poly_codes: NDArray[np.generic]
    gcp_lines: NDArray[np.float64]
    gcp_samps: NDArray[np.float64]
    gcp_lats: NDArray[np.float64]
    gcp_lons: NDArray[np.float64]
    gcp_angles: NDArray[np.float64]


def list_scene_keys(bucket: str, prefix: str, profile: str | None = None) -> list[str]:
    """Lists .nc scene object keys under an S3 prefix, sorted for a deterministic pass order."""
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".nc")
    ]
    return sorted(keys)


def download_scene(bucket: str, key: str, dest_dir: Path, profile: str | None = None) -> Path:
    """Downloads one scene object to dest_dir, returning the local path read_scene expects."""
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")
    dest_path = dest_dir / Path(key).name
    s3.download_file(bucket, key, str(dest_path))
    return dest_path


def read_scene(scene_path: Path) -> RawScene:
    """Opens the NetCDF exactly once and reads every required variable into memory;
    the returned RawScene is self-contained and needs no further file access."""
    with xr.open_dataset(scene_path, engine="netcdf4") as ds:
        sar_h, sar_w = ds["nersc_sar_primary"].shape
        return RawScene(
            scene_id=scene_path.stem,
            sar_h=sar_h,
            sar_w=sar_w,
            sar={var: ds[var].values.astype(np.float32) for var in SAR_BANDS},
            distance_map=ds["distance_map"].values.astype(np.float32),
            amsr2_raw={var: ds[var].values.astype(np.float32) for var in AMSR2_BANDS},
            era5_raw={var: ds[var].values.astype(np.float32) for var in ERA5_BANDS},
            poly_chart=ds["polygon_icechart"].values,
            poly_codes=ds["polygon_codes"].values,
            gcp_lines=ds["sar_grid_line"].values,
            gcp_samps=ds["sar_grid_sample"].values,
            gcp_lats=ds["sar_grid_latitude"].values,
            gcp_lons=ds["sar_grid_longitude"].values,
            gcp_angles=ds["sar_grid_incidenceangle"].values,
        )
