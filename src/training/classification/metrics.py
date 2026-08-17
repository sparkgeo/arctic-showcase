import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import r2_score


def sic_r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """R^2 on the 0-10 integer SIC class scale -- the primary metric, matching the
    AutoICE SIC-R2 metric directly (model_architecture.md § Downstream Classifiers).
    """
    return float(r2_score(y_true, y_pred))


def ordinal_penalty(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute class-distance between predicted and true SIC class.

    The secondary metric is specified qualitatively (weight misclassifications by
    class distance) but not pinned to one formula; mean absolute error over the
    0-10 class index is the direct reading of that description.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
