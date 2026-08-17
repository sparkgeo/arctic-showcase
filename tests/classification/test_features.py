import numpy as np
import pandas as pd
import pytest
from conftest import FakeS3

from training.classification.features import (
    ANCILLARY_COLUMNS,
    BASELINE_COLUMNS,
    CONFIG_1,
    CONFIG_2,
    CONFIG_3,
    assemble_feature_block,
    expand_feature_names,
    load_feature_matrix,
)


def _patch_rows(n: int, *, split: str, is_pure: bool, chip_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(n):
        row: dict[str, object] = {
            "split": split,
            "is_pure": is_pure,
            "label": float(i % 11),
            "chip_id": chip_id,
            "patch_token": (np.arange(1024, dtype=np.float32) + i).tolist(),
        }
        for column in [*BASELINE_COLUMNS, *ANCILLARY_COLUMNS]:
            row[column] = float(i)
        rows.append(row)
    return rows


def test_assemble_feature_block_widths_match_the_feature_contract() -> None:
    df = pd.DataFrame(_patch_rows(4, split="train", is_pure=True, chip_id="chip-0"))
    df["class_token"] = [(np.arange(1024, dtype=np.float32) + 500).tolist() for _ in range(4)]

    assert assemble_feature_block(df, CONFIG_1).shape == (4, 27)
    assert assemble_feature_block(df, CONFIG_2).shape == (4, 1046)
    assert assemble_feature_block(df, CONFIG_3).shape == (4, 2070)


def test_config3_concatenates_patch_token_then_class_token() -> None:
    df = pd.DataFrame(_patch_rows(1, split="train", is_pure=True, chip_id="chip-0"))
    df["class_token"] = [(np.arange(1024, dtype=np.float32) + 500).tolist()]

    X = assemble_feature_block(df, CONFIG_3)

    assert np.allclose(X[0, :1024], np.arange(1024, dtype=np.float32))
    assert np.allclose(X[0, 1024:2048], np.arange(1024, dtype=np.float32) + 500)


def test_expand_feature_names_matches_configuration_widths() -> None:
    assert len(expand_feature_names(CONFIG_1)) == 27
    assert len(expand_feature_names(CONFIG_2)) == 1046
    assert len(expand_feature_names(CONFIG_3)) == 2070
    assert expand_feature_names(CONFIG_2)[:3] == ["patch_token_0", "patch_token_1", "patch_token_2"]


def test_load_feature_matrix_applies_split_and_is_pure_filters(fake_s3: FakeS3) -> None:
    patch_prefix = "patch/"
    rows = (
        _patch_rows(2, split="train", is_pure=True, chip_id="chip-0")
        + _patch_rows(3, split="train", is_pure=False, chip_id="chip-0")
        + _patch_rows(1, split="validation", is_pure=True, chip_id="chip-0")
    )
    fake_s3.put_dataframe(f"{patch_prefix}scene_id=scene-a/part-0000.parquet", pd.DataFrame(rows))

    result = load_feature_matrix(
        fake_s3, "bucket", CONFIG_1, "train", is_pure_only=True, patch_table_prefix=patch_prefix
    )

    assert result.X.shape == (2, 27)
    assert result.y.tolist() == [0.0, 1.0]


def test_load_feature_matrix_drops_rows_with_no_valid_label(fake_s3: FakeS3) -> None:
    patch_prefix = "patch/"
    rows = _patch_rows(2, split="train", is_pure=True, chip_id="chip-0")
    unlabelled = _patch_rows(1, split="train", is_pure=True, chip_id="chip-0")[0]
    unlabelled["label"] = float("nan")
    fake_s3.put_dataframe(
        f"{patch_prefix}scene_id=scene-a/part-0000.parquet", pd.DataFrame([*rows, unlabelled])
    )

    result = load_feature_matrix(
        fake_s3, "bucket", CONFIG_1, "train", is_pure_only=False, patch_table_prefix=patch_prefix
    )

    assert result.X.shape == (2, 27)


def test_load_feature_matrix_concatenates_across_scene_partitions(fake_s3: FakeS3) -> None:
    patch_prefix = "patch/"
    fake_s3.put_dataframe(
        f"{patch_prefix}scene_id=scene-a/part-0000.parquet",
        pd.DataFrame(_patch_rows(2, split="train", is_pure=True, chip_id="chip-0")),
    )
    fake_s3.put_dataframe(
        f"{patch_prefix}scene_id=scene-b/part-0000.parquet",
        pd.DataFrame(_patch_rows(3, split="train", is_pure=True, chip_id="chip-1")),
    )

    result = load_feature_matrix(
        fake_s3, "bucket", CONFIG_1, "train", is_pure_only=True, patch_table_prefix=patch_prefix
    )

    assert result.X.shape == (5, 27)


def test_load_feature_matrix_returns_empty_matrix_when_nothing_matches(fake_s3: FakeS3) -> None:
    patch_prefix = "patch/"
    fake_s3.put_dataframe(
        f"{patch_prefix}scene_id=scene-a/part-0000.parquet",
        pd.DataFrame(_patch_rows(2, split="test", is_pure=True, chip_id="chip-0")),
    )

    result = load_feature_matrix(
        fake_s3,
        "bucket",
        CONFIG_1,
        "validation",
        is_pure_only=True,
        patch_table_prefix=patch_prefix,
    )

    assert result.X.shape == (0, 27)
    assert result.y.shape == (0,)


def test_load_feature_matrix_config3_joins_chip_table_on_chip_id(fake_s3: FakeS3) -> None:
    patch_prefix, chip_prefix = "patch/", "chip/"
    fake_s3.put_dataframe(
        f"{patch_prefix}scene_id=scene-a/part-0000.parquet",
        pd.DataFrame(_patch_rows(3, split="train", is_pure=True, chip_id="chip-0")),
    )
    fake_s3.put_dataframe(
        f"{chip_prefix}scene_id=scene-a/part-0000.parquet",
        pd.DataFrame(
            {
                "chip_id": ["chip-0"],
                "class_token": [(np.arange(1024, dtype=np.float32) + 500).tolist()],
            }
        ),
    )

    result = load_feature_matrix(
        fake_s3,
        "bucket",
        CONFIG_3,
        "train",
        is_pure_only=True,
        patch_table_prefix=patch_prefix,
        chip_table_prefix=chip_prefix,
    )

    assert result.X.shape == (3, 2070)
    assert np.allclose(result.X[0, 1024:2048], np.arange(1024, dtype=np.float32) + 500)


def test_load_feature_matrix_config3_raises_when_a_patch_has_no_matching_chip(
    fake_s3: FakeS3,
) -> None:
    patch_prefix, chip_prefix = "patch/", "chip/"
    fake_s3.put_dataframe(
        f"{patch_prefix}scene_id=scene-a/part-0000.parquet",
        pd.DataFrame(_patch_rows(1, split="train", is_pure=True, chip_id="chip-missing")),
    )
    fake_s3.put_dataframe(
        f"{chip_prefix}scene_id=scene-a/part-0000.parquet",
        pd.DataFrame({"chip_id": ["chip-0"], "class_token": [[0.0] * 1024]}),
    )

    with pytest.raises(AssertionError):
        load_feature_matrix(
            fake_s3,
            "bucket",
            CONFIG_3,
            "train",
            is_pure_only=True,
            patch_table_prefix=patch_prefix,
            chip_table_prefix=chip_prefix,
        )
