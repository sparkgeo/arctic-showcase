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
) -> None:
    """Writes one scene's rows as a GeoParquet partition to S3, Hive-style
    (table_prefix/scene_id=<scene_id>/part.parquet), matching the two-table,
    partition-by-scene_id layout in the pipeline architecture doc.

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
        key = f"{table_prefix}scene_id={scene_id}/part.parquet"
        s3.upload_file(str(local_path), bucket, key)
