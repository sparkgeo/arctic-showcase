import argparse
import logging

import boto3
from mypy_boto3_s3 import S3Client

from training.classification.sweep import run_sweep
from training.classification.tracking import configure_mlflow
from training.s3_paths import DEFAULT_BUCKET

DEFAULT_EXPERIMENT_NAME = "prescient-ice-b3-classifier-sweep"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the B3 feature-configuration x classifier training sweep."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument(
        "--profile",
        default=None,
        help="AWS profile name; omit to use the default credential chain "
        "(e.g. an IAM role when running in the cloud)",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="SageMaker managed MLflow tracking server ARN/URI. Omit to use "
        "whatever tracking URI mlflow is already configured with (env var or "
        "local default).",
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--is-pure-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train on pure patches only (default) vs. the full patch set. "
        "training_strategy.md documents pure-only as the natural first "
        "comparison, but the actual choice must still be recorded on the B3 issue.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Cap on scenes read per split, for a fast dev run; omit for the full corpus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_mlflow(args.mlflow_tracking_uri, args.experiment_name)

    session = boto3.Session(profile_name=args.profile)
    s3: S3Client = session.client("s3")

    results = run_sweep(s3, args.bucket, is_pure_only=args.is_pure_only, max_scenes=args.max_scenes)

    for result in sorted(results, key=lambda r: r.val_sic_r2, reverse=True):
        logger.info(
            "%s / %s: val_sic_r2=%.4f val_ordinal_penalty=%.4f (run_id=%s)",
            result.configuration,
            result.classifier,
            result.val_sic_r2,
            result.val_ordinal_penalty,
            result.run_id,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
