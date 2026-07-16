import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon

from training.data_loader.bands import GRID_SIZE, PATCH_SIZE
from training.data_loader.chip import Chip
from training.data_loader.geolocation import GcpInterpolators
from training.feature_assembly import ChipGeometry

# EPSG:3978 (NAD83 / Canada Atlas Lambert), the analytical CRS throughout this project.
# Reprojecting the corner points is an exact vector transform -- distinct from raster
# reprojection, which AI4Arctic training deliberately avoids (see pipeline architecture
# doc § Input Preparation).
_TO_EPSG3978 = Transformer.from_crs("EPSG:4326", "EPSG:3978", always_xy=True)


def build_chip_geometry(chip: Chip, gcp: GcpInterpolators) -> ChipGeometry:
    """Chip and patch footprint polygons in EPSG:3978.

    Built from one shared (GRID_SIZE + 1) x (GRID_SIZE + 1) corner grid, so
    adjacent patches meet at identical coordinates and one GCP interpolation
    call + one reprojection call covers the whole chip.
    """
    n = GRID_SIZE + 1
    corner_rows = chip.chip_row_start + np.arange(n) * PATCH_SIZE
    corner_cols = chip.chip_col_start + np.arange(n) * PATCH_SIZE
    row_grid, col_grid = np.meshgrid(corner_rows, corner_cols, indexing="ij")
    points = np.stack([row_grid.ravel(), col_grid.ravel()], axis=-1)

    lat = gcp.lat(points).reshape(n, n)
    lon = gcp.lon(points).reshape(n, n)
    x, y = _TO_EPSG3978.transform(lon, lat)

    def footprint(r0: int, r1: int, c0: int, c1: int) -> Polygon:
        return Polygon(
            [
                (x[r0, c0], y[r0, c0]),
                (x[r0, c1], y[r0, c1]),
                (x[r1, c1], y[r1, c1]),
                (x[r1, c0], y[r1, c0]),
            ]
        )

    patches = [
        footprint(pi, pi + 1, pj, pj + 1) for pi in range(GRID_SIZE) for pj in range(GRID_SIZE)
    ]
    chip_footprint = footprint(0, GRID_SIZE, 0, GRID_SIZE)

    return ChipGeometry(chip=chip_footprint, patches=patches)
