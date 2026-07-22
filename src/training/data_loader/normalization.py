import json
from collections.abc import Sequence

from mypy_boto3_s3 import S3Client


def load_stats(s3: S3Client, bucket: str, stats_key: str) -> dict[str, dict[str, float]]:
    response = s3.get_object(Bucket=bucket, Key=stats_key)
    stats: dict[str, dict[str, float]] = json.loads(response["Body"].read().decode("utf-8"))
    return stats


def load_band_means(
    s3: S3Client,
    bucket: str,
    stats_key: str,
    bands: Sequence[str],
) -> dict[str, float]:
    stats = load_stats(s3, bucket, stats_key)
    return {band: stats[band]["mean"] for band in bands if band in stats}
