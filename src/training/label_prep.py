import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from training.data_loader.bands import GRID_SIZE, PATCH_SIZE
from training.data_loader.chip import Chip

N_SIC_CLASSES = 11  # SIC classes 0-10 tenths


@dataclass(frozen=True)
class PatchLabels:
    chip_id: str
    patch_i: int
    patch_j: int
    valid_class_fraction: float
    label: float
    is_pure: bool
    frac_sic: NDArray[np.float32]  # (11,); index k = fraction of valid-class pixels in class k


def compute_patch_labels(chip: Chip) -> list[PatchLabels]:
    """
    Subdivides a chip's per-pixel chart CT data (chip.chart_ct, tenths, NaN = no valid
    class) into the 32x32 patch grid and computes per-patch label statistics: the
    per-class area-fraction vector, valid_class_fraction, the discrete label (the
    area-weighted collapse, floored rather than rounded so a mixed patch is never
    assigned more ice than it actually has -- a pure patch is the degenerate case of
    the same formula), and is_pure.
    """
    chart_ct = chip.chart_ct  # (256, 256)
    labels = []

    for pi in range(GRID_SIZE):
        for pj in range(GRID_SIZE):
            r0, r1 = pi * PATCH_SIZE, (pi + 1) * PATCH_SIZE
            c0, c1 = pj * PATCH_SIZE, (pj + 1) * PATCH_SIZE

            patch_ct = chart_ct[r0:r1, c0:c1]
            valid = ~np.isnan(patch_ct)
            n_valid = int(valid.sum())
            valid_class_fraction = n_valid / 64.0

            if n_valid > 0:
                patch_classes = np.rint(patch_ct[valid]).astype(np.int64)
                counts = np.bincount(patch_classes, minlength=N_SIC_CLASSES)[:N_SIC_CLASSES]
                frac_sic = (counts / n_valid).astype(np.float32)
                label = float(math.floor(float(np.dot(np.arange(N_SIC_CLASSES), frac_sic))))
                is_pure = bool((frac_sic == 1.0).any())
            else:
                frac_sic = np.full(N_SIC_CLASSES, np.nan, dtype=np.float32)
                label = float("nan")
                is_pure = False

            labels.append(
                PatchLabels(
                    chip_id=chip.chip_id,
                    patch_i=pi,
                    patch_j=pj,
                    valid_class_fraction=valid_class_fraction,
                    label=label,
                    is_pure=is_pure,
                    frac_sic=frac_sic,
                )
            )

    return labels
