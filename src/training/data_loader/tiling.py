from collections.abc import Iterator

from training.data_loader.bands import CHIP_SIZE


def chip_starts(scene_dim: int, chip_size: int = CHIP_SIZE) -> list[int]:
    """Regular grid of chip start offsets, with the trailing chip backward-shifted
    to align with the scene edge (covers every source pixel without padding)."""
    starts = list(range(0, scene_dim, chip_size))
    if starts[-1] + chip_size > scene_dim:
        starts[-1] = scene_dim - chip_size
    return starts


def chip_bounds(
    sar_h: int, sar_w: int, chip_size: int = CHIP_SIZE
) -> Iterator[tuple[int, int, int, int]]:
    """Yield (row_start, row_end, col_start, col_end) for every chip on the grid."""
    for row_start in chip_starts(sar_h, chip_size):
        for col_start in chip_starts(sar_w, chip_size):
            yield row_start, row_start + chip_size, col_start, col_start + chip_size
