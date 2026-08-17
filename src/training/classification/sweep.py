from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from mypy_boto3_s3 import S3Client

from training.classification.features import (
    ALL_CONFIGURATIONS,
    FeatureConfiguration,
    load_feature_matrix,
)
from training.classification.models import CLASSIFIER_BUILDERS
from training.classification.tracking import log_training_run


@dataclass(frozen=True)
class SweepResult:
    configuration: str
    classifier: str
    run_id: str
    val_sic_r2: float
    val_ordinal_penalty: float


def run_sweep(
    s3: S3Client,
    bucket: str,
    *,
    is_pure_only: bool,
    configurations: Sequence[FeatureConfiguration] = ALL_CONFIGURATIONS,
    classifier_names: Sequence[str] = tuple(CLASSIFIER_BUILDERS),
    max_scenes: int | None = None,
) -> list[SweepResult]:
    """Trains every (configuration, classifier) combination on the train split,
    evaluates on validation, and logs each run to the active MLflow experiment.

    Each configuration's train/validation feature matrices are loaded once and
    reused across classifiers, since both classifiers train on the same features.
    """
    results: list[SweepResult] = []
    for configuration in configurations:
        train = load_feature_matrix(
            s3, bucket, configuration, "train", is_pure_only=is_pure_only, max_scenes=max_scenes
        )
        val = load_feature_matrix(
            s3,
            bucket,
            configuration,
            "validation",
            is_pure_only=is_pure_only,
            max_scenes=max_scenes,
        )
        y_train = train.y.astype(np.int64)
        y_val = val.y.astype(np.int64)

        for classifier_name in classifier_names:
            model = CLASSIFIER_BUILDERS[classifier_name]()
            run_id, val_r2, val_penalty = log_training_run(
                configuration.name,
                classifier_name,
                model,
                train.feature_names,
                train.X,
                y_train,
                val.X,
                y_val,
            )
            results.append(
                SweepResult(configuration.name, classifier_name, run_id, val_r2, val_penalty)
            )
    return results
