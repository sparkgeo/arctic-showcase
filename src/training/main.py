import argparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from mypy_boto3_s3 import S3Client

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
from training.s3_paths import CHIP_TABLE_PREFIX, DEFAULT_BUCKET, PATCH_TABLE_PREFIX

S3_TRAIN_PREFIX = "training_data/ai4arctic/raw_train/"
S3_TEST_PREFIX = "training_data/ai4arctic/raw_test/"
STATS_KEY = "training_data/ai4arctic/statistics/dataset_stats.json"
# Local-test cap on the number of scenes processed. None (or a non-positive
# --max-scenes) runs the full corpus.
DEFAULT_MAX_SCENES: int | None = 2

_REPO_ROOT = Path(__file__).resolve().parents[2]
CLAY_CHECKPOINT_PATH = _REPO_ROOT / "clay-v1.5.ckpt"
CLAY_CHECKPOINT_S3_PREFIX = "model_files/clay-v1.5.ckpt"
CLAY_METADATA_PATH = _REPO_ROOT / "configs" / "metadata.yaml"

CHIP_LOG_INTERVAL = 50  # log progress every N chips within a scene
# Flush + clear the patch-row buffer every N chips
PATCH_FLUSH_INTERVAL = 50

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneRef:
    key: str
    split: str


def scene_already_processed(
    s3: S3Client, bucket: str, chip_table_prefix: str, scene_id: str
) -> bool:
    """True if scene_id's chip_table partition is already in S3. chip_table is the
    last write in a scene's processing (see main()), so its presence means the
    scene's GPU encoding and both tables already completed on a prior run."""
    prefix = f"{chip_table_prefix}scene_id={scene_id}/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return response["KeyCount"] > 0


def list_all_scenes(
    s3: S3Client, bucket: str, train_prefix: str, test_prefix: str
) -> list[SceneRef]:
    """All 533 scenes, each carrying its split assignment: raw_train -> "train",
    raw_test -> "test".

    The design docs describe the 20 test scenes' labels as withheld from
    AI4Arctic's original distribution and requiring a separate post-challenge
    join. Verified directly against this bucket's raw_test data (polygon_codes
    has real, non-sentinel CT values; build_chart_ct yields a sane multi-class
    distribution): that join already happened before this copy was uploaded, so
    reading raw_test through load_scene/compute_patch_labels needs no extra step.
    """
    train_keys = list_scene_keys(s3, bucket, train_prefix)
    test_keys = list_scene_keys(s3, bucket, test_prefix)
    return [SceneRef(key=key, split="train") for key in train_keys] + [
        SceneRef(key=key, split="test") for key in test_keys
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the B2 training feature-assembly harness.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument(
        "--profile",
        default=None,
        help="AWS profile name; omit to use the default credential chain "
        "(e.g. an IAM role when running in the cloud)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=DEFAULT_MAX_SCENES,
        help="Cap on the number of scenes processed; pass a non-positive value to "
        "process the full corpus",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bucket = args.bucket
    max_scenes = args.max_scenes if args.max_scenes is not None and args.max_scenes > 0 else None

    session = boto3.Session(profile_name=args.profile)
    s3: S3Client = session.client("s3")

    band_means = load_band_means(s3, bucket, STATS_KEY, ALL_BANDS)
    stats = load_stats(s3, bucket, STATS_KEY)

    device = select_device()
    sar_meta = ensure_sentinel1_ew_entry(CLAY_METADATA_PATH, stats)

    # Retrieve the ClayMAE checkpoint from S3 if it's not already present locally.
    if not CLAY_CHECKPOINT_PATH.exists():
        s3.download_file(bucket, CLAY_CHECKPOINT_S3_PREFIX, str(CLAY_CHECKPOINT_PATH))
    module = load_clay_module(CLAY_CHECKPOINT_PATH, CLAY_METADATA_PATH, device)

    all_scenes = list_all_scenes(s3, bucket, S3_TRAIN_PREFIX, S3_TEST_PREFIX)
    if max_scenes is not None:
        all_scenes = all_scenes[:max_scenes]
    logger.info("Processing %d scenes", len(all_scenes))

    failed_scenes: list[str] = []

    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)

        for scene_idx, scene in enumerate(all_scenes, start=1):
            scene_id = Path(scene.key).stem
            if scene_already_processed(s3, bucket, CHIP_TABLE_PREFIX, scene_id):
                logger.info(
                    "[%d/%d] %s: already processed, skipping", scene_idx, len(all_scenes), scene.key
                )
                continue

            scene_path: Path | None = None
            try:
                scene_patch_rows: list[dict[str, Any]] = []
                scene_chip_rows: list[dict[str, Any]] = []

                logger.info(
                    "[%d/%d] %s (split=%s): downloading",
                    scene_idx,
                    len(all_scenes),
                    scene.key,
                    scene.split,
                )

                # download + load stand in for the pseudocode's single load_chips(scene) --
                # scenes live in S3, and load_scene needs a local NetCDF path.
                scene_path = download_scene(s3, bucket, scene.key, scratch_dir)
                loaded_scene = load_scene(scene_path, band_means)

                chip_count = 0
                total_patch_rows = 0
                patch_part = 0
                for chip in load_chips(loaded_scene):
                    embeddings = encode_chip(
                        chip, module, sar_meta, device
                    )  # B2.2 -- embeddings only
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

                    chip_count += 1
                    if chip_count % CHIP_LOG_INTERVAL == 0:
                        logger.info(
                            "[%d/%d] %s: %d chips done",
                            scene_idx,
                            len(all_scenes),
                            scene.key,
                            chip_count,
                        )

                    if chip_count % PATCH_FLUSH_INTERVAL == 0:
                        total_patch_rows += len(scene_patch_rows)
                        write_partition(
                            s3,
                            bucket,
                            PATCH_TABLE_PREFIX,
                            loaded_scene.scene_id,
                            scene_patch_rows,
                            part=patch_part,
                        )
                        patch_part += 1
                        scene_patch_rows.clear()

                scene_path.unlink()
                scene_path = None

                # flush whatever's left (a partial batch, or a whole small scene),
                # then the chip table -- chip rows are ~1024x smaller than patch
                # rows (no patch_token), so buffering them for the whole scene is fine.
                total_patch_rows += len(scene_patch_rows)
                write_partition(
                    s3,
                    bucket,
                    PATCH_TABLE_PREFIX,
                    loaded_scene.scene_id,
                    scene_patch_rows,
                    part=patch_part,
                )
                write_partition(
                    s3, bucket, CHIP_TABLE_PREFIX, loaded_scene.scene_id, scene_chip_rows
                )

                logger.info(
                    "[%d/%d] %s: done -- %d chips, %d patch rows, %d chip rows",
                    scene_idx,
                    len(all_scenes),
                    scene.key,
                    chip_count,
                    total_patch_rows,
                    len(scene_chip_rows),
                )
            except Exception:
                logger.exception(
                    "[%d/%d] %s: failed, skipping", scene_idx, len(all_scenes), scene.key
                )
                failed_scenes.append(scene.key)
            finally:
                if scene_path is not None:
                    scene_path.unlink()

    if failed_scenes:
        logger.warning(
            "%d/%d scene(s) failed and were skipped: %s",
            len(failed_scenes),
            len(all_scenes),
            failed_scenes,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
