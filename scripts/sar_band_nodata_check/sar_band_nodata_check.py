import argparse
import csv
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
from mypy_boto3_s3 import S3Client

from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS
from training.data_loader.scene_reader import list_scene_keys, read_scene
from training.data_loader.valid_mask import compute_valid_mask

BUCKET = "prescient-ice-data"
S3_PREFIX = "training_data/ai4arctic/raw_train/"

# One representative variable per ancillary source: nodata is a sensor/product-level
# flag shared across a source's channels, so checking one stands in for all of them.
AMSR2_REPRESENTATIVE_BAND = AMSR2_BANDS[0]
ERA5_REPRESENTATIVE_BAND = ERA5_BANDS[0]


def compute_scene_asymmetry(s3: S3Client, bucket: str, key: str) -> dict | None:
    """Per scene: HH-alone, HV-alone, and combined (both-bands) valid pixel fractions
    over non-land pixels, plus the one-directional mismatch counts between bands.
    Also reports the raw (native-grid, pre-resample) valid pixel fraction for one
    representative AMSR2 and ERA5 variable each.

    Reuses compute_valid_mask (the actual production nodata union) for the combined
    figure, rather than re-deriving it, so this can never silently drift from what
    load_scene actually does.
    """
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local = Path(tmpdir) / key.split("/")[-1]
            s3.download_file(bucket, key, str(local))
            raw = read_scene(local)
    except Exception as e:
        print(f"  ERR {key}: {e}")
        return None

    primary = raw.sar["nersc_sar_primary"]
    secondary = raw.sar["nersc_sar_secondary"]
    distance_map = raw.distance_map

    not_land = distance_map != 0
    n_not_land = int(not_land.sum())
    if n_not_land == 0:
        print(f"  SKIP {key}: no non-land pixels")
        return None

    valid_primary_alone = ~np.isnan(primary) & not_land
    valid_secondary_alone = ~np.isnan(secondary) & not_land
    valid_combined = compute_valid_mask(primary, secondary, distance_map)

    primary_only = int((valid_primary_alone & ~valid_secondary_alone).sum())
    secondary_only = int((valid_secondary_alone & ~valid_primary_alone).sum())

    # No land mask: AMSR2/ERA5 are physically valid over land and open water alike
    # (ancillary.py), so their only nodata is the sensor/product fill value, already
    # decoded to NaN by scene_reader.read_scene.
    amsr2_var = raw.amsr2_raw[AMSR2_REPRESENTATIVE_BAND]
    era5_var = raw.era5_raw[ERA5_REPRESENTATIVE_BAND]
    n_amsr2_pixels = amsr2_var.size
    n_era5_pixels = era5_var.size
    amsr2_valid_pixels = int((~np.isnan(amsr2_var)).sum())
    era5_valid_pixels = int((~np.isnan(era5_var)).sum())

    elapsed = time.time() - t0
    print(f"  OK  {key.split('/')[-1][:60]}  ({elapsed:.1f}s)")

    return {
        "scene": key.split("/")[-1],
        "n_not_land_pixels": n_not_land,
        "primary_valid_pixels": int(valid_primary_alone.sum()),
        "secondary_valid_pixels": int(valid_secondary_alone.sum()),
        "combined_valid_pixels": int(valid_combined.sum()),
        "primary_only_pixels": primary_only,
        "secondary_only_pixels": secondary_only,
        "n_amsr2_pixels": n_amsr2_pixels,
        "amsr2_valid_pixels": amsr2_valid_pixels,
        "n_era5_pixels": n_era5_pixels,
        "era5_valid_pixels": era5_valid_pixels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare HH-alone/HV-alone/combined SAR valid-pixel fractions across "
        "training scenes, to quantify how often and how badly HH/HV nodata footprints diverge. "
        "Also reports the AMSR2/ERA5 valid-pixel fraction on their native grids, using one "
        "representative variable per source."
    )
    parser.add_argument("--bucket", default=BUCKET, help="S3 bucket name")
    parser.add_argument("--prefix", default=S3_PREFIX, help="S3 key prefix for .nc scene files")
    parser.add_argument("--profile", default=None, help="AWS CLI profile name")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download threads")
    parser.add_argument("--csv", default="sar_band_nodata_asymmetry.csv")
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of worst-mismatch scenes to print"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N scenes (sorted order) instead of the whole corpus",
    )
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile)
    s3 = session.client("s3")

    print(f"Listing scenes at s3://{args.bucket}/{args.prefix} ...")
    keys = list_scene_keys(s3, args.bucket, args.prefix)
    print(f"Found {len(keys)} scenes")
    if args.limit is not None:
        keys = keys[: args.limit]
        print(f"Limiting to first {len(keys)} scenes")
    print()

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(compute_scene_asymmetry, s3, args.bucket, k): k for k in keys}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)

    csv_path = Path(args.csv)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} scene row(s) to {csv_path}\n")

    if not rows:
        raise SystemExit("No scenes processed successfully.")

    total_not_land = sum(r["n_not_land_pixels"] for r in rows)
    total_primary = sum(r["primary_valid_pixels"] for r in rows)
    total_secondary = sum(r["secondary_valid_pixels"] for r in rows)
    total_combined = sum(r["combined_valid_pixels"] for r in rows)
    total_primary_only = sum(r["primary_only_pixels"] for r in rows)
    total_secondary_only = sum(r["secondary_only_pixels"] for r in rows)
    total_amsr2_pixels = sum(r["n_amsr2_pixels"] for r in rows)
    total_amsr2_valid = sum(r["amsr2_valid_pixels"] for r in rows)
    total_era5_pixels = sum(r["n_era5_pixels"] for r in rows)
    total_era5_valid = sum(r["era5_valid_pixels"] for r in rows)

    print("Corpus-wide (pixel-weighted across all non-land pixels in all scenes):")
    print(f"  HH-alone valid fraction:       {total_primary / total_not_land:.6f}")
    print(f"  HV-alone valid fraction:       {total_secondary / total_not_land:.6f}")
    print(f"  Combined valid fraction:       {total_combined / total_not_land:.6f}")
    print(f"  HH-valid-but-HV-not (pixels):  {total_primary_only:,}")
    print(f"  HV-valid-but-HH-not (pixels):  {total_secondary_only:,}")
    print(
        f"\nCorpus-wide (pixel-weighted, native grid, '{AMSR2_REPRESENTATIVE_BAND}' "
        "as representative of all AMSR2 channels):"
    )
    print(f"  AMSR2 valid fraction:          {total_amsr2_valid / total_amsr2_pixels:.6f}")
    print(
        f"\nCorpus-wide (pixel-weighted, native grid, '{ERA5_REPRESENTATIVE_BAND}' "
        "as representative of all ERA5 channels):"
    )
    print(f"  ERA5 valid fraction:           {total_era5_valid / total_era5_pixels:.6f}")

    for r in rows:
        r["mismatch_frac"] = (r["primary_only_pixels"] + r["secondary_only_pixels"]) / r[
            "n_not_land_pixels"
        ]

    worst = sorted(rows, key=lambda r: r["mismatch_frac"], reverse=True)[: args.top_n]
    print(f"\nTop {len(worst)} scene(s) by HH/HV mismatch fraction:")
    for r in worst:
        print(
            f"  {r['mismatch_frac']:.6f}  {r['scene']}  "
            f"(HH-only={r['primary_only_pixels']:,}, HV-only={r['secondary_only_pixels']:,}, "
            f"non-land={r['n_not_land_pixels']:,})"
        )
