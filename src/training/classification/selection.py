import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
from mypy_boto3_s3 import S3Client

from training.classification.features import CONFIGURATIONS_BY_NAME, load_feature_matrix
from training.classification.metrics import ordinal_penalty, sic_r2
from training.s3_paths import MODEL_ARTIFACT_PREFIX

logger = logging.getLogger(__name__)


def score_on_test(
    s3: S3Client,
    bucket: str,
    run_id: str,
    configuration_name: str,
    *,
    is_pure_only: bool,
    max_scenes: int | None = None,
) -> tuple[float, float]:
    """Scores an already-trained, MLflow-logged model once on the 20 held-out test
    scenes. Call this only after the model-selection decision is made and
    documented on the issue -- the test split must never inform selection
    (prescient_ice_training_strategy.md § Dataset Splits).
    """
    configuration = CONFIGURATIONS_BY_NAME[configuration_name]
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    test = load_feature_matrix(
        s3, bucket, configuration, "test", is_pure_only=is_pure_only, max_scenes=max_scenes
    )
    y_true = test.y.astype(np.int64)
    y_pred = model.predict(test.X)

    test_r2 = sic_r2(y_true, y_pred)
    test_penalty = ordinal_penalty(y_true, y_pred)

    with mlflow.start_run(run_name=f"test-score-{configuration_name}"):
        mlflow.set_tag("selected_run_id", run_id)
        mlflow.log_metrics({"test_sic_r2": test_r2, "test_ordinal_penalty": test_penalty})

    logger.info(
        "test_sic_r2=%.4f test_ordinal_penalty=%.4f (source run_id=%s)",
        test_r2,
        test_penalty,
        run_id,
    )
    return test_r2, test_penalty


def register_selected_model(
    s3: S3Client,
    bucket: str,
    run_id: str,
    registered_model_name: str = "prescient-ice-sic-classifier",
    model_artifact_prefix: str = MODEL_ARTIFACT_PREFIX,
) -> str:
    """Pins the selected run's model in the MLflow model registry and copies its
    artefact to the agreed S3 path for A9 to read directly.
    """
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, registered_model_name)
    s3_prefix = f"{model_artifact_prefix}{registered_model_name}/v{registered.version}/"

    with tempfile.TemporaryDirectory() as scratch:
        local_dir = Path(mlflow.artifacts.download_artifacts(model_uri, dst_path=scratch))
        for local_path in local_dir.rglob("*"):
            if local_path.is_file():
                key = f"{s3_prefix}{local_path.relative_to(local_dir)}"
                s3.upload_file(str(local_path), bucket, key)

    return f"s3://{bucket}/{s3_prefix}"
