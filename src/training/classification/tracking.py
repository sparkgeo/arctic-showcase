import subprocess
from typing import Any

import mlflow
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import classification_report, confusion_matrix

from training.classification.metrics import ordinal_penalty, sic_r2
from training.label_prep import N_SIC_CLASSES

_SIC_CLASS_LABELS = list(range(N_SIC_CLASSES))


def configure_mlflow(tracking_uri: str | None, experiment_name: str) -> None:
    """Points the MLflow client at the SageMaker managed tracking server.

    Enabling the tracking server itself is a one-time SageMaker Studio console step
    (prescient_ice_training_strategy.md § Experiment Tracking); this only configures
    the client once that server exists and its ARN/URI is known.
    """
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_training_run(
    configuration_name: str,
    classifier_name: str,
    model: Any,
    feature_names: list[str],
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    X_val: NDArray[np.float32],
    y_val: NDArray[np.int64],
) -> tuple[str, float, float]:
    """Fits `model` and logs params, train/val metrics, the val confusion matrix,
    per-class precision/recall, feature importances, git commit, and the model
    artefact to the active MLflow run. Returns (run_id, val_sic_r2,
    val_ordinal_penalty) for model selection.
    """
    log_model = (
        mlflow.sklearn.log_model if classifier_name == "random_forest" else mlflow.xgboost.log_model
    )

    with mlflow.start_run(run_name=f"{configuration_name}-{classifier_name}") as run:
        mlflow.log_params(
            {
                "feature_configuration": configuration_name,
                "classifier": classifier_name,
                "n_features": X_train.shape[1],
                "n_train_rows": X_train.shape[0],
                **model.get_params(),
            }
        )
        mlflow.set_tag("git_commit", _git_commit())

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        val_r2 = sic_r2(y_val, val_pred)
        val_penalty = ordinal_penalty(y_val, val_pred)
        mlflow.log_metrics(
            {
                "train_sic_r2": sic_r2(y_train, train_pred),
                "train_ordinal_penalty": ordinal_penalty(y_train, train_pred),
                "val_sic_r2": val_r2,
                "val_ordinal_penalty": val_penalty,
            }
        )

        mlflow.log_dict(
            {
                "labels": _SIC_CLASS_LABELS,
                "matrix": confusion_matrix(y_val, val_pred, labels=_SIC_CLASS_LABELS).tolist(),
            },
            "val_confusion_matrix.json",
        )
        mlflow.log_dict(
            classification_report(
                y_val, val_pred, labels=_SIC_CLASS_LABELS, output_dict=True, zero_division=0
            ),
            "val_classification_report.json",
        )

        if hasattr(model, "feature_importances_"):
            mlflow.log_dict(
                dict(zip(feature_names, model.feature_importances_.tolist(), strict=True)),
                "feature_importances.json",
            )

        log_model(model, artifact_path="model")

        return run.info.run_id, val_r2, val_penalty
