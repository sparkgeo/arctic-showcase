import numpy as np
import pytest

from training.classification.metrics import ordinal_penalty, sic_r2


def test_sic_r2_is_one_for_perfect_predictions() -> None:
    y = np.array([0.0, 5.0, 10.0, 3.0])
    assert sic_r2(y, y) == 1.0


def test_ordinal_penalty_is_mean_absolute_class_distance() -> None:
    y_true = np.array([0.0, 5.0, 10.0])
    y_pred = np.array([1.0, 5.0, 7.0])
    assert ordinal_penalty(y_true, y_pred) == pytest.approx((1 + 0 + 3) / 3)


def test_ordinal_penalty_is_zero_for_perfect_predictions() -> None:
    y = np.array([0.0, 5.0, 10.0])
    assert ordinal_penalty(y, y) == 0.0
