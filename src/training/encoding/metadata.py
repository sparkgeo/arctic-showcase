from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# HH index 0, HV index 1 -- must match the SAR band ordering everywhere else.
_SAR_BAND_KEYS = ("nersc_sar_primary", "nersc_sar_secondary")
_WAVELENGTHS = [5.405, 5.405]  # nominal C-band, consistent with Clay's built-in SAR entries
_GSD = 40.0
_ENTRY_NAME = "sentinel-1-ew"


@dataclass(frozen=True)
class SarMetadata:
    band_names: list[str]
    gsd: float
    mean: list[float]
    std: list[float]
    wavelengths: list[float]


def build_sentinel1_ew_entry(stats: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {
        "band_names": ["hh", "hv"],
        "gsd": _GSD,
        "wavelengths": _WAVELENGTHS,
        "mean": [stats[band]["mean"] for band in _SAR_BAND_KEYS],
        "std": [stats[band]["std"] for band in _SAR_BAND_KEYS],
    }


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open() as f:
        metadata: dict[str, Any] = yaml.safe_load(f)
    return metadata


def save_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
    with metadata_path.open("w") as f:
        yaml.dump(metadata, f, default_flow_style=False)


def sar_metadata_from_entry(entry: dict[str, Any]) -> SarMetadata:
    return SarMetadata(
        band_names=list(entry["band_names"]),
        gsd=float(entry["gsd"]),
        mean=list(entry["mean"]),
        std=list(entry["std"]),
        wavelengths=list(entry["wavelengths"]),
    )


def ensure_sentinel1_ew_entry(
    metadata_path: Path, stats: dict[str, dict[str, float]]
) -> SarMetadata:
    """Idempotent: only builds and writes the entry if it isn't already committed."""
    metadata = load_metadata(metadata_path)
    if _ENTRY_NAME not in metadata:
        metadata[_ENTRY_NAME] = build_sentinel1_ew_entry(stats)
        save_metadata(metadata_path, metadata)
    return sar_metadata_from_entry(metadata[_ENTRY_NAME])
