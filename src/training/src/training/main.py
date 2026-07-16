import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.data_loader import (
    ALL_BANDS,
    download_scene,
    list_scene_keys,
    load_band_means,
    load_scene,
    load_stats,
)
from training.data_loader import yield_chips as load_chips
from training.encoding import (
    encode_chip,
    ensure_sentinel1_ew_entry,
    load_clay_module,
    select_device,
)
from training.feature_assembly import assemble_rows
from training.geometry import build_chip_geometry
from training.label_prep import compute_patch_labels as prepare_labels
from training.parquet_writer import write_partition
from training.patch_features import compute_patch_features as extract_patch_features

BUCKET = "prescient-ice-data"
S3_TRAIN_PREFIX = "training_data/ai4arctic/raw_train/"
S3_TEST_PREFIX = "training_data/ai4arctic/raw_test/"
STATS_KEY = "training_data/ai4arctic/statistics/dataset_stats.json"
# TODO: placeholder output location -- not yet a documented/agreed destination.
PATCH_TABLE_PREFIX = "training_data/ai4arctic/features/patch_table/"
CHIP_TABLE_PREFIX = "training_data/ai4arctic/features/chip_table/"
PROFILE = "spk_data"

# TODO: change to obtain from S3 instead?
_REPO_ROOT = Path(__file__).resolve().parents[4]
CLAY_CHECKPOINT_PATH = _REPO_ROOT / "clay-v1.5.ckpt"
CLAY_METADATA_PATH = _REPO_ROOT / "configs" / "metadata.yaml"


@dataclass(frozen=True)
class SceneRef:
    key: str
    split: str


def list_all_scenes(
    bucket: str, train_prefix: str, test_prefix: str, profile: str | None
) -> list[SceneRef]:
    """All 533 scenes, each carrying its split assignment: raw_train -> "train",
    raw_test -> "test".

    TODO: the 20 test scenes' labels were withheld from AI4Arctic's original
    distribution and released separately post-challenge -- that label file still
    needs to be joined in at load time (B2.1) for compute_patch_labels to produce
    real values on the test split; reading raw_test through load_scene alone does
    not do this join.
    """
    train_keys = list_scene_keys(bucket, train_prefix, profile=profile)
    test_keys = list_scene_keys(bucket, test_prefix, profile=profile)
    return [SceneRef(key=key, split="train") for key in train_keys] + [
        SceneRef(key=key, split="test") for key in test_keys
    ]


def main() -> None:
    band_means = load_band_means(BUCKET, STATS_KEY, ALL_BANDS, profile=PROFILE)
    stats = load_stats(BUCKET, STATS_KEY, profile=PROFILE)

    device = select_device()
    sar_meta = ensure_sentinel1_ew_entry(CLAY_METADATA_PATH, stats)
    module = load_clay_module(CLAY_CHECKPOINT_PATH, CLAY_METADATA_PATH, device)

    all_scenes = list_all_scenes(BUCKET, S3_TRAIN_PREFIX, S3_TEST_PREFIX, PROFILE)

    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)

        for scene in all_scenes:  # all 533, each carrying its split assignment
            scene_patch_rows: list[dict[str, Any]] = []
            scene_chip_rows: list[dict[str, Any]] = []

            # download + load stand in for the pseudocode's single load_chips(scene) --
            # scenes live in S3, and load_scene needs a local NetCDF path.
            scene_path = download_scene(BUCKET, scene.key, scratch_dir, profile=PROFILE)
            loaded_scene = load_scene(scene_path, band_means)

            for chip in load_chips(loaded_scene):
                embeddings = encode_chip(chip, module, sar_meta, device)  # B2.2 -- embeddings only
                patch_features = extract_patch_features(chip)  # B2.3
                labels = prepare_labels(chip)  # B2.4
                geometry = build_chip_geometry(chip, loaded_scene.gcp)

                patch_rows, chip_row = assemble_rows(  # B2.5
                    embeddings, patch_features, labels, geometry, chip.scene_id
                )
                for row in patch_rows:
                    row["split"] = scene.split
                chip_row["split"] = scene.split

                scene_patch_rows.extend(patch_rows)
                scene_chip_rows.append(chip_row)

            scene_path.unlink()

            # durable write, then free -- one partition per table per scene, so a
            # failure mid-run leaves every already-flushed scene persisted.
            write_partition(
                BUCKET, PATCH_TABLE_PREFIX, loaded_scene.scene_id, scene_patch_rows, PROFILE
            )
            write_partition(
                BUCKET, CHIP_TABLE_PREFIX, loaded_scene.scene_id, scene_chip_rows, PROFILE
            )


if __name__ == "__main__":
    main()
