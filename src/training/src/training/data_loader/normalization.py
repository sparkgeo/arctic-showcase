import json
from collections.abc import Sequence

import boto3


def load_band_means(
    bucket: str,
    stats_key: str,
    bands: Sequence[str],
    profile: str | None = None,
) -> dict[str, float]:
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")
    response = s3.get_object(Bucket=bucket, Key=stats_key)
    stats = json.loads(response["Body"].read().decode("utf-8"))
    return {band: stats[band]["mean"] for band in bands if band in stats}
