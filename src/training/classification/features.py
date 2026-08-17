from dataclasses import dataclass

import numpy as np
import pandas as pd
from mypy_boto3_s3 import S3Client
from numpy.typing import NDArray

from training.classification.tables import list_partition_scene_ids, read_table_partition
from training.data_loader.bands import ERA5_BANDS
from training.feature_assembly import AMSR2_CONTRACT_COLUMNS
from training.s3_paths import CHIP_TABLE_PREFIX, PATCH_TABLE_PREFIX

# Blocks 3-5 of the feature contract (model_architecture.md § Feature Contract),
# appended after Clay's tokens in every configuration.
BASELINE_COLUMNS: list[str] = ["hh_mean", "hv_mean", "hh_std", "hv_std", "hv_hh_ratio"]
ANCILLARY_COLUMNS: list[str] = [*AMSR2_CONTRACT_COLUMNS, *ERA5_BANDS, "distance_to_land", "ia_mean"]

# Fixed-length list columns and their stored width -- everything else is a scalar column.
_LIST_COLUMN_WIDTHS: dict[str, int] = {"patch_token": 1024, "class_token": 1024}


@dataclass(frozen=True)
class FeatureConfiguration:
    name: str
    raw_columns: list[str]  # patch-table columns to read, in contract block order (unexpanded)
    needs_chip_join: bool
    expected_dim: int


CONFIG_1 = FeatureConfiguration(
    name="config1",
    raw_columns=[*BASELINE_COLUMNS, *ANCILLARY_COLUMNS],
    needs_chip_join=False,
    expected_dim=27,
)
CONFIG_2 = FeatureConfiguration(
    name="config2",
    raw_columns=["patch_token", *ANCILLARY_COLUMNS],
    needs_chip_join=False,
    expected_dim=1046,
)
CONFIG_3 = FeatureConfiguration(
    name="config3",
    raw_columns=["patch_token", "class_token", *ANCILLARY_COLUMNS],
    needs_chip_join=True,
    expected_dim=2070,
)
ALL_CONFIGURATIONS: tuple[FeatureConfiguration, ...] = (CONFIG_1, CONFIG_2, CONFIG_3)
CONFIGURATIONS_BY_NAME: dict[str, FeatureConfiguration] = {c.name: c for c in ALL_CONFIGURATIONS}


@dataclass(frozen=True)
class FeatureMatrix:
    X: NDArray[np.float32]
    y: NDArray[np.float32]
    feature_names: list[str]


def _expand_column(df: pd.DataFrame, column: str) -> NDArray[np.float32]:
    width = _LIST_COLUMN_WIDTHS.get(column)
    if width is None:
        scalar_column: NDArray[np.float32] = df[column].to_numpy(dtype=np.float32).reshape(-1, 1)
        return scalar_column
    values = np.stack(df[column].to_numpy(), axis=0).astype(np.float32)
    assert values.shape == (len(df), width), f"{column} has drifted from its contract width"
    return values


def expand_feature_names(configuration: FeatureConfiguration) -> list[str]:
    names: list[str] = []
    for column in configuration.raw_columns:
        width = _LIST_COLUMN_WIDTHS.get(column)
        names.extend([column] if width is None else [f"{column}_{i}" for i in range(width)])
    return names


def assemble_feature_block(
    df: pd.DataFrame, configuration: FeatureConfiguration
) -> NDArray[np.float32]:
    """Concatenates a configuration's raw columns, in contract block order, into one
    dense float32 matrix -- list columns (patch_token, class_token) expand to their
    fixed width; scalar columns become a single column. Pure tabular assembly, no
    computation, mirroring feature_assembly.assemble_rows's role on the write side.
    """
    X = np.concatenate([_expand_column(df, column) for column in configuration.raw_columns], axis=1)
    assert X.shape[1] == configuration.expected_dim, (
        f"{configuration.name} produced {X.shape[1]} columns, expected {configuration.expected_dim}"
    )
    return X


def _apply_read_time_filters(df: pd.DataFrame, split: str, is_pure_only: bool) -> pd.DataFrame:
    filtered = df[df["split"] == split]
    if is_pure_only:
        filtered = filtered[filtered["is_pure"]]
    # label is NaN only for patches with zero valid-class pixels (label_prep.py);
    # such patches carry no usable training target.
    return filtered[filtered["label"].notna()]


def load_feature_matrix(
    s3: S3Client,
    bucket: str,
    configuration: FeatureConfiguration,
    split: str,
    *,
    is_pure_only: bool,
    patch_table_prefix: str = PATCH_TABLE_PREFIX,
    chip_table_prefix: str = CHIP_TABLE_PREFIX,
    max_scenes: int | None = None,
) -> FeatureMatrix:
    """Assembles one feature configuration's design matrix for one split.

    Reads the patch table scene-by-scene, applies the split and is_pure_only
    read-time filters, and -- for Configuration 3 -- joins each scene's chip-table
    partition on chip_id before projecting the configuration's columns. No stored
    column is re-derived; filtering and the join are the only operations here.
    """
    patch_columns = list(
        dict.fromkeys(["split", "is_pure", "label", "chip_id", *configuration.raw_columns])
    )
    patch_columns = [c for c in patch_columns if c != "class_token"]

    scene_ids = list_partition_scene_ids(s3, bucket, patch_table_prefix)
    if max_scenes is not None:
        scene_ids = scene_ids[:max_scenes]

    X_parts: list[NDArray[np.float32]] = []
    y_parts: list[NDArray[np.float32]] = []
    for scene_id in scene_ids:
        patch_df = read_table_partition(s3, bucket, patch_table_prefix, scene_id, patch_columns)
        patch_df = _apply_read_time_filters(patch_df, split, is_pure_only)
        if patch_df.empty:
            continue

        if configuration.needs_chip_join:
            chip_df = read_table_partition(
                s3, bucket, chip_table_prefix, scene_id, ["chip_id", "class_token"]
            )
            patch_df = patch_df.merge(chip_df, on="chip_id", how="left", validate="many_to_one")
            assert patch_df["class_token"].notna().all(), (
                f"scene {scene_id}: patch rows with no matching chip_id in the chip table"
            )

        X_parts.append(assemble_feature_block(patch_df, configuration))
        y_parts.append(patch_df["label"].to_numpy(dtype=np.float32))

    feature_names = expand_feature_names(configuration)
    if not X_parts:
        return FeatureMatrix(
            X=np.empty((0, configuration.expected_dim), dtype=np.float32),
            y=np.empty((0,), dtype=np.float32),
            feature_names=feature_names,
        )
    return FeatureMatrix(
        X=np.concatenate(X_parts, axis=0),
        y=np.concatenate(y_parts, axis=0),
        feature_names=feature_names,
    )
