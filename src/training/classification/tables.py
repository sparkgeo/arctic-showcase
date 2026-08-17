from io import BytesIO

import pandas as pd
from mypy_boto3_s3 import S3Client


def list_partition_scene_ids(s3: S3Client, bucket: str, table_prefix: str) -> list[str]:
    """Scene IDs present in a Hive-partitioned (scene_id=<id>/) GeoParquet table."""
    paginator = s3.get_paginator("list_objects_v2")
    scene_ids = []
    for page in paginator.paginate(Bucket=bucket, Prefix=table_prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            partition = common_prefix["Prefix"][len(table_prefix) :].rstrip("/")
            scene_ids.append(partition.removeprefix("scene_id="))
    return sorted(scene_ids)


def read_table_partition(
    s3: S3Client, bucket: str, table_prefix: str, scene_id: str, columns: list[str]
) -> pd.DataFrame:
    """Reads and concatenates every part file in one scene's partition, projecting
    only `columns` -- geometry and any other unrequested column never enters the
    frame, matching the feature contract's "geometry projects out at load time"
    requirement.
    """
    partition_prefix = f"{table_prefix}scene_id={scene_id}/"
    paginator = s3.get_paginator("list_objects_v2")
    keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket, Prefix=partition_prefix)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    parts = [
        pd.read_parquet(
            BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read()), columns=columns
        )
        for key in keys
    ]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)
