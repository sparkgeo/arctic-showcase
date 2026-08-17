import argparse
import logging

import boto3
from mypy_boto3_s3 import S3Client

from training.classification.selection import register_selected_model, score_on_test
from training.classification.tracking import configure_mlflow
from training.s3_paths import DEFAULT_BUCKET
from training.train_main import DEFAULT_EXPERIMENT_NAME

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a manually-selected B3 run on the held-out test scenes, "
        "then pin it in MLflow and serialise its artefact to S3 for A9. Run this only "
        "after the model-selection decision from the sweep has been made and "
        "documented on the B3 issue -- it is never invoked automatically."
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-id", required=True, help="MLflow run_id of the selected model.")
    parser.add_argument("--configuration", required=True, choices=["config1", "config2", "config3"])
    parser.add_argument(
        "--is-pure-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Must match the is_pure_only setting the selected run was trained with.",
    )
    parser.add_argument("--registered-model-name", default="prescient-ice-sic-classifier")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_mlflow(args.mlflow_tracking_uri, args.experiment_name)

    session = boto3.Session(profile_name=args.profile)
    s3: S3Client = session.client("s3")

    test_r2, test_penalty = score_on_test(
        s3, args.bucket, args.run_id, args.configuration, is_pure_only=args.is_pure_only
    )
    logger.info("headline test_sic_r2=%.4f test_ordinal_penalty=%.4f", test_r2, test_penalty)

    s3_uri = register_selected_model(s3, args.bucket, args.run_id, args.registered_model_name)
    logger.info("registered %s, artefact at %s", args.registered_model_name, s3_uri)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
