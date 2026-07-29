import numpy as np

from training.data_loader.resampling import resample_to_sar


def test_resample_to_sar_upsamples_to_the_target_shape() -> None:
    arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    result = resample_to_sar(arr, target_h=4, target_w=4)

    assert result.shape == (4, 4)
    assert result.dtype == np.float32


def test_resample_to_sar_does_not_smear_nan_into_valid_neighbours() -> None:
    """A plain scipy.ndimage.zoom would blend the NaN into every output pixel whose
    interpolation kernel touches it, corrupting otherwise-uniform valid data nearby.
    A 2x2 gap (rather than a single isolated cell) guarantees some output pixels'
    interpolation stencil is fully within the gap, so it must survive as NaN too."""
    arr = np.full((6, 6), 100.0, dtype=np.float32)
    arr[2:4, 2:4] = np.nan  # a genuine gap

    result = resample_to_sar(arr, target_h=12, target_w=12)

    valid = ~np.isnan(result)
    assert np.isnan(result).any()  # the gap itself survives
    assert np.allclose(result[valid], 100.0)  # nothing nearby gets pulled off 100


def test_resample_to_sar_returns_all_nan_when_source_is_all_nan() -> None:
    arr = np.full((2, 2), np.nan, dtype=np.float32)

    result = resample_to_sar(arr, target_h=4, target_w=4)

    assert np.isnan(result).all()
