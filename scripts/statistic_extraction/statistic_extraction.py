import argparse
import csv
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import xarray as xr
from mypy_boto3_s3 import S3Client

BUCKET = "prescient-ice-data"
S3_PREFIX = "training_data/ai4arctic/raw_train/"

VARIABLES = [
    "nersc_sar_primary",
    "nersc_sar_secondary",
    "btemp_6_9h",
    "btemp_6_9v",
    "btemp_18_7h",
    "btemp_18_7v",
    "btemp_36_5h",
    "btemp_36_5v",
    "btemp_89_0h",
    "btemp_89_0v",
    "u10m_rotated",
    "v10m_rotated",
    "t2m",
    "skt",
    "tcwv",
    "tclw",
]


def list_scene_keys(s3: S3Client, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".nc"):
                keys.append(obj["Key"])
    return keys


def compute_scene_stats(s3: S3Client, bucket: str, key: str) -> dict[str, dict] | None:
    """Download one scene from S3 and return per-variable (n, mean, M2).

    Uses a two-pass approach (mean first, then squared deviations) so variance
    is computed without large cancellation regardless of mean magnitude.
    """
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = Path(tmpdir) / key.split("/")[-1]
            s3.download_file(bucket, key, str(local))
            stats: dict[str, dict] = {}
            with xr.open_dataset(local) as ds:
                for var in VARIABLES:
                    if var not in ds:
                        continue
                    arr = ds[var].values.astype(np.float64).flatten()
                    # Validate the data by removing NaNs and sentinel values (255, 2)
                    valid = arr[~np.isnan(arr)]
                    valid = valid[valid != 255]
                    valid = valid[valid != 2]
                    if len(valid) == 0:
                        continue
                    # Compute mean and M2 (sum of squared deviations) for variance calculation
                    mean = float(np.mean(valid))
                    M2 = float(np.sum((valid - mean) ** 2))
                    stats[var] = {
                        "n": len(valid),
                        "mean": mean,
                        "M2": M2,
                        "std": float(np.std(valid)),
                        "min": float(np.min(valid)),
                        "max": float(np.max(valid)),
                    }
        # Logging the elapsed time for processing the scene
        elapsed = time.time() - t0
        print(f"  OK  {key.split('/')[-1][:60]}  ({elapsed:.1f}s)")
        return stats
    except Exception as e:
        print(f"  ERR {key}: {e}")
        return None


def merge(a: dict, b: dict) -> dict:
    """Merge two (n, mean, M2) accumulators"""
    n = a["n"] + b["n"]
    delta = b["mean"] - a["mean"]
    mean = a["mean"] + delta * b["n"] / n
    M2 = a["M2"] + b["M2"] + delta**2 * a["n"] * b["n"] / n
    return {"n": n, "mean": mean, "M2": M2}


def finalise(accumulated: dict[str, dict]) -> dict[str, dict]:
    return {
        var: {
            "mean": round(s["mean"], 6),
            "std": round(float(np.sqrt(s["M2"] / s["n"])), 6),
            "n_pixels": s["n"],
        }
        for var, s in accumulated.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-variable mean/std over all training scenes."
    )
    parser.add_argument("--bucket", default=BUCKET, help="S3 bucket name")
    parser.add_argument("--prefix", default=S3_PREFIX, help="S3 key prefix for .nc scene files")
    parser.add_argument("--profile", default=None, help="AWS CLI profile name")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download threads")
    parser.add_argument("--output", default="dataset_stats.json")
    parser.add_argument("--csv", default="scene_stats.csv", help="Per-scene CSV output path")
    args = parser.parse_args()

    # Change this when running in Sagemaker
    session = boto3.Session(profile_name="spk_data")
    s3 = session.client("s3")

    print(f"Listing scenes at s3://{args.bucket}/{args.prefix} ...")
    keys = list_scene_keys(s3, args.bucket, args.prefix)
    print(f"Found {len(keys)} scenes\n")

    total_start = time.time()
    accumulated: dict[str, dict] = {}

    # Write per-scene stats to CSV while accumulating overall stats
    csv_path = Path(args.csv)
    csv_exists = csv_path.exists()
    csv_file = csv_path.open("a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["scene", "variable", "mean", "std", "min", "max", "n_pixels"])

    # Pooling the scene processing to speed up the computation
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(compute_scene_stats, s3, args.bucket, k): k for k in keys}
        for future in as_completed(futures):
            key = futures[future]
            result = future.result()
            if result is None:
                continue
            scene_name = key.split("/")[-1]
            for var, s in result.items():
                csv_writer.writerow(
                    [scene_name, var, s["mean"], s["std"], s["min"], s["max"], s["n"]]
                )
                csv_file.flush()
                if var not in accumulated:
                    accumulated[var] = {"n": 0, "mean": 0.0, "M2": 0.0}
                accumulated[var] = merge(accumulated[var], s)

    csv_file.close()

    final = finalise(accumulated)
    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)

    total_elapsed = time.time() - total_start
    print(f"\nTotal wall time: {total_elapsed:.1f}s  →  {args.output}\n")

    print(f"  {'variable':25s}  {'mean':>10}  {'std':>10}  {'n_pixels':>14}")
    print("  " + "-" * 65)
    for var, s in final.items():
        print(f"  {var:25s}  {s['mean']:10.4f}  {s['std']:10.4f}  {s['n_pixels']:14,}")
