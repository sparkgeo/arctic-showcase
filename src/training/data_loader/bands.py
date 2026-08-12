CHIP_SIZE = 256  # pixels
PATCH_SIZE = 8  # pixels per patch side
N_PATCHES = (CHIP_SIZE // PATCH_SIZE) ** 2  # 1024 patches per chip
GRID_SIZE = 32

SAR_BANDS = [
    "nersc_sar_primary",  # HH
    "nersc_sar_secondary",  # HV
]

AMSR2_BANDS = [
    "btemp_6_9h",
    "btemp_6_9v",
    "btemp_7_3h",
    "btemp_7_3v",
    "btemp_10_7h",
    "btemp_10_7v",
    "btemp_18_7h",
    "btemp_18_7v",
    "btemp_23_8h",
    "btemp_23_8v",
    "btemp_36_5h",
    "btemp_36_5v",
    "btemp_89_0h",
    "btemp_89_0v",
]

ERA5_BANDS = [
    "u10m_rotated",
    "v10m_rotated",
    "t2m",
    "skt",
    "tcwv",
    "tclw",
]

ANCILLARY_BANDS = ["sar_grid_incidenceangle", "distance_map"]

ALL_BANDS = SAR_BANDS + AMSR2_BANDS + ERA5_BANDS
