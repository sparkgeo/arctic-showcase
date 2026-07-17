import tempfile
from pathlib import Path
from typing import Any

import boto3
import geopandas as gpd

CRS = "EPSG:3978"


def write_partition(
    bucket: str,
    table_prefix: str,
    scene_id: str,
    rows: list[dict[str, Any]],
    profile: str | None = None,
    part: int = 0,
) -> None:
    """Writes one batch of a scene's rows as a GeoParquet file to S3, Hive-style
    (table_prefix/scene_id=<scene_id>/part-<part>.parquet), matching the two-table,
    partition-by-scene_id layout in the pipeline architecture doc.

    A scene's rows may be written across several calls with increasing part
    numbers (see main.py's per-chip flush) to bound peak memory -- multiple
    files under one scene_id partition is normal for a Parquet dataset.

    Writes nothing for an empty row list -- a scene contributing no rows
    (e.g. all chips skipped) leaves no partition rather than an empty file.
    """
    if not rows:
        return

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)

    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")

    with tempfile.TemporaryDirectory() as scratch:
        local_path = Path(scratch) / "part.parquet"
        gdf.to_parquet(local_path)
        key = f"{table_prefix}scene_id={scene_id}/part-{part:04d}.parquet"
        s3.upload_file(str(local_path), bucket, key)
