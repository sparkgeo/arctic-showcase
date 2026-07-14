import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.data_loader import (
    ALL_BANDS,
    Chip,
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
from training.feature_assembly import ChipGeometry, assemble_rows
from training.label_prep import compute_patch_labels as prepare_labels
from training.patch_features import compute_patch_features as extract_patch_features

BUCKET = "prescient-ice-data"
S3_PREFIX = "training_data/ai4arctic/raw_train/"
STATS_KEY = "training_data/ai4arctic/statistics/dataset_stats.json"
PROFILE = "spk_data"

# TODO: real locations once the checkpoint / committed Clay metadata.yaml are decided.
CLAY_CHECKPOINT_PATH = Path("TODO_clay_checkpoint.ckpt")
CLAY_METADATA_PATH = Path("TODO_clay_metadata.yaml")


@dataclass(frozen=True)
class SceneRef:
    key: str
    split: str


def list_all_scenes(bucket: str, prefix: str, profile: str | None) -> list[SceneRef]:
    """All 533 scenes, each carrying its split assignment.

    TODO: only the raw_train prefix is listed here, and every scene is tagged
    "train" -- the real train/validation manifest and the separate raw_test
    prefix (prescient_ice_training_strategy.md § Dataset Splits) aren't wired in.
    """
    keys = list_scene_keys(bucket, prefix, profile=profile)
    return [SceneRef(key=key, split="train") for key in keys]


def chip_geometry(chip: Chip) -> ChipGeometry:
    """TODO: chip/patch footprint polygons from the GCP grid aren't built anywhere yet."""
    raise NotImplementedError("chip/patch geometry construction is not yet implemented")


def flush_parquet_partition(
    scene_id: str, buffer: list[tuple[list[dict[str, Any]], dict[str, Any]]]
) -> None:
    """TODO: no GeoParquet writer exists yet -- this is B2.6's durable write."""
    raise NotImplementedError("GeoParquet partition write is not yet implemented")


def main() -> None:
    band_means = load_band_means(BUCKET, STATS_KEY, ALL_BANDS, profile=PROFILE)
    stats = load_stats(BUCKET, STATS_KEY, profile=PROFILE)

    device = select_device()
    sar_meta = ensure_sentinel1_ew_entry(CLAY_METADATA_PATH, stats)
    module = load_clay_module(CLAY_CHECKPOINT_PATH, CLAY_METADATA_PATH, device)

    all_scenes = list_all_scenes(BUCKET, S3_PREFIX, PROFILE)

    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)

        for scene in all_scenes:  # all 533, each carrying its split assignment
            buffer: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

            # download + load stand in for the pseudocode's single load_chips(scene) --
            # scenes live in S3, and load_scene needs a local NetCDF path.
            scene_path = download_scene(BUCKET, scene.key, scratch_dir, profile=PROFILE)
            loaded_scene = load_scene(scene_path, band_means)

            for chip in load_chips(loaded_scene):
                embeddings = encode_chip(chip, module, sar_meta, device)  # B2.2 -- embeddings only
                patch_features = extract_patch_features(chip)  # B2.3
                labels = prepare_labels(chip)  # B2.4
                geometry = chip_geometry(chip)

                patch_rows, chip_row = assemble_rows(  # B2.5
                    embeddings, patch_features, labels, geometry, chip.scene_id
                )
                for row in patch_rows:
                    row["split"] = scene.split
                chip_row["split"] = scene.split

                buffer.append((patch_rows, chip_row))

            scene_path.unlink()
            flush_parquet_partition(loaded_scene.scene_id, buffer)  # durable write, then free


if __name__ == "__main__":
    main()
