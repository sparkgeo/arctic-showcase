import numpy as np
from training.data_loader.labels import build_chart_ct, parse_ct_tenths


def test_build_chart_ct_maps_polygon_ids_to_concentration() -> None:
    assert parse_ct_tenths("80") == 8.0

    poly_codes = np.array(["poly_id;CT", "1;80", "2;-9"])
    poly_chart = np.array([[1.0, 2.0], [1.0, np.nan]], dtype=np.float64)

    chart_ct = build_chart_ct(poly_chart, poly_codes)

    assert chart_ct[0, 0] == 8.0
    assert chart_ct[1, 0] == 8.0
    assert np.isnan(chart_ct[0, 1])  # polygon 2 has the -9 fill CT code
    assert np.isnan(chart_ct[1, 1])  # fill polygon id (NaN in poly_chart)
