import numpy as np

from training.data_loader.labels import build_chart_ct, parse_ct_tenths


def test_parse_ct_tenths_resolves_every_sic_lookup_code() -> None:
    assert parse_ct_tenths("80") == 8.0
    assert parse_ct_tenths("0") == 0.0
    assert parse_ct_tenths("55") == 0.0
    assert parse_ct_tenths("91") == 10.0  # "9+/10"
    assert parse_ct_tenths("92") == 10.0  # fast ice
    assert parse_ct_tenths("-9") is None  # SIGRID-3 fill code
    assert parse_ct_tenths("99") is None  # ICECHART_UNKNOWN

    # range codes resolve each endpoint through the same table before averaging
    assert parse_ct_tenths("50-70") == 6.0
    assert parse_ct_tenths("90-92") == 9.5


def test_build_chart_ct_maps_polygon_ids_to_concentration() -> None:
    poly_codes = np.array(["poly_id;CT", "1;80", "2;-9", "3;91"])
    poly_chart = np.array([[1.0, 2.0], [1.0, 3.0]], dtype=np.float64)

    chart_ct = build_chart_ct(poly_chart, poly_codes)

    assert chart_ct[0, 0] == 8.0
    assert chart_ct[1, 0] == 8.0
    assert np.isnan(chart_ct[0, 1])  # polygon 2 has the -9 fill CT code
    assert chart_ct[1, 1] == 10.0  # polygon 3 has the 91 ("9+/10") code, not a sentinel
