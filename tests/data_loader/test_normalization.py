import json
from unittest.mock import MagicMock

from training.data_loader.normalization import load_band_means, load_stats


def test_load_band_means_reads_stats_from_s3_and_extracts_the_mean() -> None:
    stats = {
        "nersc_sar_primary": {"mean": -15.0, "std": 3.0},
        "nersc_sar_secondary": {"mean": -22.0, "std": 4.0},
    }
    body = MagicMock()
    body.read.return_value = json.dumps(stats).encode("utf-8")
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": body}

    assert load_stats(mock_s3, "my-bucket", "stats.json") == stats

    means = load_band_means(
        mock_s3, "my-bucket", "stats.json", ["nersc_sar_primary", "unused_band"]
    )

    assert means == {"nersc_sar_primary": -15.0}
    mock_s3.get_object.assert_called_with(Bucket="my-bucket", Key="stats.json")
