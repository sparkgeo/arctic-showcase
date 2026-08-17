from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sklearn.ensemble import RandomForestClassifier

from training.label_prep import N_SIC_CLASSES

if TYPE_CHECKING:
    from xgboost import XGBClassifier

RANDOM_STATE = 42


def build_random_forest(**overrides: Any) -> RandomForestClassifier:
    params: dict[str, Any] = {
        "n_estimators": 300,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    params.update(overrides)
    return RandomForestClassifier(**params)


def build_xgboost(**overrides: Any) -> "XGBClassifier":
    # Imported lazily so environments without a working xgboost native
    # library (e.g. macOS without Homebrew's libomp) can still use the
    # random_forest path and the rest of this package.
    from xgboost import XGBClassifier

    params: dict[str, Any] = {
        "n_estimators": 300,
        "objective": "multi:softmax",
        "num_class": N_SIC_CLASSES,
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    params.update(overrides)
    return XGBClassifier(**params)


CLASSIFIER_BUILDERS: dict[str, Callable[..., Any]] = {
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
}
