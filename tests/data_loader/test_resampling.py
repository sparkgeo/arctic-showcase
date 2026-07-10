import numpy as np
from training.data_loader.resampling import resample_to_sar


def test_resample_to_sar_upsamples_to_the_target_shape() -> None:
    arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    result = resample_to_sar(arr, target_h=4, target_w=4)

    assert result.shape == (4, 4)
    assert result.dtype == np.float32
