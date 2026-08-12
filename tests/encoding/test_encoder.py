from pathlib import Path

import numpy as np
import pytest

from training.data_loader.chip import Chip
from training.encoding.encoder import encode_chip
from training.encoding.metadata import load_metadata, sar_metadata_from_entry
from training.encoding.model import load_clay_module, select_device

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_PATH = _REPO_ROOT / "clay-v1.5.ckpt"
_METADATA_PATH = _REPO_ROOT / "configs" / "metadata.yaml"

pytestmark = pytest.mark.skipif(
    not _CHECKPOINT_PATH.exists(),
    reason="requires the local clay-v1.5.ckpt checkpoint (downloaded from S3, not committed)",
)


def _make_chip() -> Chip:
    sar = np.random.default_rng(0).normal(-18.0, 5.0, size=(2, 256, 256)).astype(np.float32)
    return Chip(
        sar=sar,
        amsr2=np.zeros((14, 256, 256), dtype=np.float32),
        era5=np.zeros((6, 256, 256), dtype=np.float32),
        distance_map=np.zeros((256, 256), dtype=np.uint8),
        incidence_angle=np.zeros((256, 256), dtype=np.float32),
        valid_mask=np.ones((256, 256), dtype=np.bool_),
        chart_ct=np.zeros((256, 256), dtype=np.float32),
        chip_row_start=0,
        chip_col_start=0,
        time_encoding=np.zeros(4, dtype=np.float32),
        latlon_encoding=np.zeros(4, dtype=np.float32),
        centroid_lat=60.0,
        centroid_lon=-80.0,
        scene_id="scene",
        chip_id="scene_r00000_c00000",
    )


def test_encode_chip_returns_1024_dim_patch_and_class_tokens() -> None:
    """Empirically confirms Clay v1.5 large returns [B, 1025, 1024] on a real
    EW HH/HV chip, resolving the embedding-dimension open question rather than
    assuming it from model_size="large" alone."""
    metadata = load_metadata(_METADATA_PATH)
    sar_meta = sar_metadata_from_entry(metadata["sentinel-1-ew"])
    device = select_device()
    module = load_clay_module(_CHECKPOINT_PATH, _METADATA_PATH, device)

    chip = _make_chip()
    patch_tokens, cls_token = encode_chip(chip, module, sar_meta, device)

    assert patch_tokens.shape == (1, 1024, 32, 32)
    assert cls_token.shape == (1, 1024)
