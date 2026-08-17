from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from training.data_loader.bands import AMSR2_BANDS, ERA5_BANDS, GRID_SIZE, N_PATCHES
from training.label_prep import N_SIC_CLASSES, PatchLabels
from training.patch_features import PatchFeatures

# Source variable paired with their feature-contract column names.
_AMSR2_COLUMN_MAP: list[tuple[str, str]] = [
    ("btemp_6_9h", "btemp_6.9h"),
    ("btemp_6_9v", "btemp_6.9v"),
    ("btemp_7_3h", "btemp_7.3h"),
    ("btemp_7_3v", "btemp_7.3v"),
    ("btemp_10_7h", "btemp_10.7h"),
    ("btemp_10_7v", "btemp_10.7v"),
    ("btemp_18_7h", "btemp_18.7h"),
    ("btemp_18_7v", "btemp_18.7v"),
    ("btemp_23_8h", "btemp_23.8h"),
    ("btemp_23_8v", "btemp_23.8v"),
    ("btemp_36_5h", "btemp_36.5h"),
    ("btemp_36_5v", "btemp_36.5v"),
    ("btemp_89_0h", "btemp_89.0h"),
    ("btemp_89_0v", "btemp_89.0v"),
]
assert [source for source, _ in _AMSR2_COLUMN_MAP] == AMSR2_BANDS, (
    "AMSR2 column map has drifted from bands.AMSR2_BANDS"
)

# Feature-contract AMSR2 column names, in contract order -- the single source of
# truth for downstream consumers (e.g. classification.features) building the same
# column order without re-deriving it.
AMSR2_CONTRACT_COLUMNS: list[str] = [contract for _, contract in _AMSR2_COLUMN_MAP]

# Clay's encoder output: (patch_tokens, class_token) as returned by
# encoding.encode_chip -- shapes (1, 1024, 32, 32) and (1, 1024).
ChipEmbeddings = tuple[NDArray[np.float32], NDArray[np.float32]]


@dataclass(frozen=True)
class ChipGeometry:
    """Chip and patch footprints in EPSG:3978.

    patches must be ordered row-major (patch_i, patch_j) to match
    patch_features and labels.
    """

    chip: BaseGeometry
    patches: list[BaseGeometry]


def assemble_rows(
    embeddings: ChipEmbeddings,
    patch_features: list[PatchFeatures],
    labels: list[PatchLabels],
    geometry: ChipGeometry,
    scene_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assemble patch-grain and chip-grain rows in feature-contract order.

    patch_features and labels must be ordered row-major (patch_i, patch_j),
    matching the (h, w) unflattening of Clay's patch-token grid.
    """
    patch_tokens, class_token = embeddings
    assert patch_tokens.shape == (1, 1024, GRID_SIZE, GRID_SIZE)
    assert class_token.shape == (1, 1024)
    assert len(patch_features) == len(labels) == len(geometry.patches) == N_PATCHES

    chip_id = patch_features[0].chip_id
    assert all(f.chip_id == chip_id for f in patch_features)
    assert all(label.chip_id == chip_id for label in labels)

    patch_rows: list[dict[str, Any]] = []
    for idx, (features, label) in enumerate(zip(patch_features, labels)):
        pi, pj = idx // GRID_SIZE, idx % GRID_SIZE
        assert features.patch_i == label.patch_i == pi
        assert features.patch_j == label.patch_j == pj
        assert set(features.amsr2) == set(AMSR2_BANDS)
        assert set(features.era5) == set(ERA5_BANDS)

        row: dict[str, Any] = {
            "hh_mean": features.hh_mean,
            "hv_mean": features.hv_mean,
            "hh_std": features.hh_std,
            "hv_std": features.hv_std,
            "hv_hh_ratio": features.hv_hh_ratio,
            "patch_token": patch_tokens[0, :, pi, pj].tolist(),
        }
        for source_name, contract_name in _AMSR2_COLUMN_MAP:
            row[contract_name] = features.amsr2[source_name]
        for era5_name in ERA5_BANDS:
            row[era5_name] = features.era5[era5_name]
        row["distance_to_land"] = features.distance
        row["ia_mean"] = features.ia_mean
        for k in range(N_SIC_CLASSES):
            row[f"frac_sic{k}"] = float(label.frac_sic[k])
        row["valid_class_fraction"] = label.valid_class_fraction
        row["label"] = label.label
        row["is_pure"] = label.is_pure
        row["valid_fraction"] = features.valid_fraction
        row["chip_id"] = chip_id
        row["scene_id"] = scene_id
        row["geometry"] = geometry.patches[idx]
        patch_rows.append(row)

    chip_row: dict[str, Any] = {
        "class_token": class_token[0, :].tolist(),
        "chip_id": chip_id,
        "scene_id": scene_id,
        "geometry": geometry.chip,
    }

    return patch_rows, chip_row
