import numpy as np
from numpy.typing import NDArray

# AI4ArcticSeaIceChallenge/utils.py's SIC_LOOKUP
_SIC_LOOKUP: dict[int, float] = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    55: 0.0,
    10: 1.0,
    20: 2.0,
    30: 3.0,
    40: 4.0,
    50: 5.0,
    60: 6.0,
    70: 7.0,
    80: 8.0,
    90: 9.0,
    91: 10.0,
    92: 10.0,
}


def parse_ct_tenths(ct_str: str) -> float | None:
    """Convert a raw SIGRID-3 CT string to tenths (0-10 float) via the AI4Arctic
    SIC_LOOKUP table, or None for the -9 fill code or any code absent from the table."""
    s = ct_str.strip()
    if s == "-9":
        return None
    if "-" in s[1:]:  # range code e.g. '50-70' -> midpoint of each endpoint's tenths
        lo, _, hi = s.partition("-")
        lo_tenths = _SIC_LOOKUP.get(int(float(lo)))
        hi_tenths = _SIC_LOOKUP.get(int(float(hi)))
        if lo_tenths is None or hi_tenths is None:
            return None
        return (lo_tenths + hi_tenths) / 2.0
    return _SIC_LOOKUP.get(int(float(s)))


def build_chart_ct(
    poly_chart: NDArray[np.generic], poly_codes: NDArray[np.generic]
) -> NDArray[np.float32]:
    header = str(poly_codes[0]).split(";")
    ct_col = header.index("CT")

    ct_lookup: dict[int, float] = {}
    for row in poly_codes[1:]:
        parts = str(row).split(";")
        value = parse_ct_tenths(parts[ct_col])
        if value is not None:
            ct_lookup[int(parts[0])] = value

    # polygon_icechart may be float64 (NaN at fill) or uint16 (65535 at fill)
    if np.issubdtype(poly_chart.dtype, np.floating):
        valid_polygon = ~np.isnan(poly_chart)
        poly_ids_int = np.where(valid_polygon, poly_chart.astype(np.int32), 0)
    else:
        valid_polygon = poly_chart != 65535
        poly_ids_int = poly_chart.astype(np.int32)

    chart_ct_full: NDArray[np.float32] = np.full(poly_chart.shape, np.nan, dtype=np.float32)
    if valid_polygon.any():
        max_pid = int(poly_ids_int[valid_polygon].max())
        ct_vec = np.full(max_pid + 1, np.nan, dtype=np.float32)
        for pid, value in ct_lookup.items():
            if pid <= max_pid:
                ct_vec[pid] = value
        chart_ct_full[valid_polygon] = ct_vec[poly_ids_int[valid_polygon]]

    return chart_ct_full
