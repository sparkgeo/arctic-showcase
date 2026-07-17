import argparse
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from mypy_boto3_s3 import S3Client
from shapely.geometry import Point
from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS, GRID_SIZE
from training.feature_assembly import ChipGeometry, assemble_rows
from training.label_prep import N_SIC_CLASSES, PatchLabels
from training.patch_features import PatchFeatures

BUCKET = "prescient-ice-data"
PATCH_TABLE_PREFIX = "training_data/ai4arctic/features/patch_table/"
CHIP_TABLE_PREFIX = "training_data/ai4arctic/features/chip_table/"
PROFILE = "spk_data"

N_PATCHES = GRID_SIZE * GRID_SIZE


def expected_columns() -> tuple[list[str], list[str]]:
    """Derives the canonical patch/chip column order from assemble_rows itself
    (synthetic single-chip call) rather than hand-duplicating the feature
    contract here, so this check can't silently drift from the real schema.
    "split" is added back in -- assemble_rows doesn't own it, the harness does.
    """
    patch_features = [
        PatchFeatures(
            chip_id="synthetic",
            patch_i=i // GRID_SIZE,
            patch_j=i % GRID_SIZE,
            valid_fraction=1.0,
            hh_mean=0.0,
            hv_mean=0.0,
            hh_std=0.0,
            hv_std=0.0,
            hv_hh_ratio=0.0,
            ia_mean=30.0,
            distance=1.0,
            amsr2=dict.fromkeys(AMSR2_BANDS, 200.0),
            era5=dict.fromkeys(ERA5_BANDS, 0.0),
        )
        for i in range(N_PATCHES)
    ]
    labels = [
        PatchLabels(
            chip_id="synthetic",
            patch_i=i // GRID_SIZE,
            patch_j=i % GRID_SIZE,
            valid_class_fraction=1.0,
            label=0.0,
            is_pure=True,
            frac_sic=np.zeros(N_SIC_CLASSES, dtype=np.float32),
        )
        for i in range(N_PATCHES)
    ]
    footprint = Point(0, 0).buffer(1)
    geometry = ChipGeometry(chip=footprint, patches=[footprint] * N_PATCHES)
    embeddings = (
        np.zeros((1, 1024, GRID_SIZE, GRID_SIZE), dtype=np.float32),
        np.zeros((1, 1024), dtype=np.float32),
    )
    patch_rows, chip_row = assemble_rows(embeddings, patch_features, labels, geometry, "synthetic")
    return [*patch_rows[0].keys(), "split"], [*chip_row.keys(), "split"]


def list_partitions(s3: S3Client, bucket: str, table_prefix: str) -> dict[str, list[str]]:
    """scene_id -> part-file keys under that scene's Hive-style partition."""
    print(f"Listing s3://{bucket}/{table_prefix} ...")
    partitions: dict[str, list[str]] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=table_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "scene_id=" not in key:
                continue
            scene_id = key.split("scene_id=", 1)[1].split("/", 1)[0]
            partitions.setdefault(scene_id, []).append(key)
    n_parts = sum(len(v) for v in partitions.values())
    print(f"  found {len(partitions)} scene(s), {n_parts} part file(s)")
    return partitions


def read_partition(s3: S3Client, bucket: str, keys: list[str]) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as scratch:
        frames = []
        for i, key in enumerate(keys, start=1):
            print(f"    downloading part {i}/{len(keys)}: {key.split('/')[-1]}")
            local_path = Path(scratch) / key.split("/")[-1]
            s3.download_file(bucket, key, str(local_path))
            frames.append(pd.read_parquet(local_path))
        df = pd.concat(frames, ignore_index=True)
        print(f"    read {len(df)} rows from {len(keys)} part file(s)")
        return df


def validate_patch_table(df: pd.DataFrame, expected_cols: list[str], sample_size: int) -> list[str]:
    print(f"    validating patch table ({len(df)} rows, {len(df.columns)} columns)")
    errors = []

    print("      checking column schema...")
    if list(df.columns) != expected_cols:
        missing = set(expected_cols) - set(df.columns)
        extra = set(df.columns) - set(expected_cols)
        errors.append(f"column mismatch: missing={missing or None}, extra={extra or None}")

    print("      checking patch rows per chip_id...")
    counts = df.groupby("chip_id").size()
    if (counts > N_PATCHES).any():
        errors.append(f"{(counts > N_PATCHES).sum()} chip_id(s) exceed {N_PATCHES} patch rows")

    print("      checking valid_fraction / valid_class_fraction bounds...")
    for col in ("valid_fraction", "valid_class_fraction"):
        if col in df and ((df[col] < 0) | (df[col] > 1)).any():
            errors.append(f"{col} outside [0, 1]")

    sample = df.sample(min(sample_size, len(df)), random_state=0) if len(df) else df
    print(f"      checking patch_token length on a sample of {len(sample)} rows...")
    if "patch_token" in sample and not sample["patch_token"].apply(len).eq(1024).all():
        errors.append("patch_token length != 1024 in sample")

    print("      checking frac_sic0..frac_sic10 sum to 1.0...")
    frac_cols = [f"frac_sic{k}" for k in range(N_SIC_CLASSES)]
    if all(c in df.columns for c in frac_cols):
        if "valid_class_fraction" in sample:
            labelled = sample[sample["valid_class_fraction"] > 0]
        else:
            labelled = sample
        sums = labelled[frac_cols].sum(axis=1)
        if len(sums) and not np.allclose(sums, 1.0, atol=1e-3):
            errors.append("frac_sic0..frac_sic10 don't sum to 1.0 for a labelled sample row")

    print(f"    patch table: {len(errors)} issue(s) found")
    return errors


def validate_chip_table(df: pd.DataFrame, expected_cols: list[str], sample_size: int) -> list[str]:
    print(f"    validating chip table ({len(df)} rows, {len(df.columns)} columns)")
    errors = []

    print("      checking column schema...")
    if list(df.columns) != expected_cols:
        missing = set(expected_cols) - set(df.columns)
        extra = set(df.columns) - set(expected_cols)
        errors.append(f"column mismatch: missing={missing or None}, extra={extra or None}")

    print("      checking chip_id uniqueness...")
    if df["chip_id"].duplicated().any():
        errors.append("duplicate chip_id in chip table")

    sample = df.sample(min(sample_size, len(df)), random_state=0) if len(df) else df
    print(f"      checking class_token length on a sample of {len(sample)} rows...")
    if "class_token" in sample and not sample["class_token"].apply(len).eq(1024).all():
        errors.append("class_token length != 1024 in sample")

    print(f"    chip table: {len(errors)} issue(s) found")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the B2.6-assembled patch/chip GeoParquet tables against the "
        "feature contract."
    )
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--patch-prefix", default=PATCH_TABLE_PREFIX)
    parser.add_argument("--chip-prefix", default=CHIP_TABLE_PREFIX)
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile)
    s3 = session.client("s3")

    print("Deriving expected schema from assemble_rows...")
    expected_patch_cols, expected_chip_cols = expected_columns()
    print(f"  {len(expected_patch_cols)} patch columns, {len(expected_chip_cols)} chip columns\n")

    patch_partitions = list_partitions(s3, args.bucket, args.patch_prefix)
    chip_partitions = list_partitions(s3, args.bucket, args.chip_prefix)

    scene_ids = sorted(set(patch_partitions) | set(chip_partitions))
    print(f"\nValidating {len(scene_ids)} scene(s)\n")

    all_chip_ids_patch: set[str] = set()
    all_chip_ids_chip: set[str] = set()
    any_failures = False

    for scene_id in scene_ids:
        print(f"scene_id={scene_id}")
        if scene_id not in patch_partitions:
            print("  FAIL  no patch_table partition")
            any_failures = True
            continue
        if scene_id not in chip_partitions:
            print("  FAIL  no chip_table partition")
            any_failures = True
            continue

        print(f"  reading patch_table partition ({len(patch_partitions[scene_id])} file(s))...")
        patch_df = read_partition(s3, args.bucket, patch_partitions[scene_id])
        print(f"  reading chip_table partition ({len(chip_partitions[scene_id])} file(s))...")
        chip_df = read_partition(s3, args.bucket, chip_partitions[scene_id])
        all_chip_ids_patch.update(patch_df["chip_id"].unique())
        all_chip_ids_chip.update(chip_df["chip_id"].unique())

        errors = validate_patch_table(
            patch_df, expected_patch_cols, args.sample_size
        ) + validate_chip_table(chip_df, expected_chip_cols, args.sample_size)

        if errors:
            any_failures = True
            for e in errors:
                print(f"  FAIL  {e}")
        else:
            print(f"  OK    {len(patch_df)} patch rows, {len(chip_df)} chip rows")
        print()

    print("Checking chip_id join coverage across both tables...")
    unmatched = all_chip_ids_patch ^ all_chip_ids_chip
    if unmatched:
        any_failures = True
        print(
            f"\nFAIL  {len(unmatched)} chip_id(s) present in only one table (breaks the "
            f"Configuration 3 join)"
        )
    else:
        print(f"\nOK    all {len(all_chip_ids_patch)} chip_id(s) join cleanly across both tables")

    print("\nRESULT:", "FAIL" if any_failures else "PASS")


if __name__ == "__main__":
    main()
