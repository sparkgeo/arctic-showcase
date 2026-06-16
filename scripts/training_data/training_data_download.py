import argparse
import tempfile
import time
from pathlib import Path

import boto3
import botocore.exceptions
import requests
from mypy_boto3_s3 import S3Client
from tqdm import tqdm

MAX_RETRIES = 5
RETRY_BACKOFF = 2.0  # seconds, doubles each attempt

DATASETS = {
    "rtt_train": {
        "source_url": "https://data.dtu.dk/articles/dataset/Ready-To-Train_AI4Arctic_Sea_Ice_Challenge_Dataset/21316608",
        "s3_prefix": "training_data/ai4arctic/rtt_train/",
    },
    "rtt_test": {
        "source_url": "https://data.dtu.dk/articles/dataset/Ready-To-Train_AI4Arctic_Sea_Ice_Challenge_Test_Dataset/21762830",
        "s3_prefix": "training_data/ai4arctic/rtt_test/",
    },
    "raw_train": {
        "source_url": "https://data.dtu.dk/articles/dataset/Raw_AI4Arctic_Sea_Ice_Challenge_Dataset/21284967",
        "s3_prefix": "training_data/ai4arctic/raw_train/",
    },
    "raw_test": {
        "source_url": "https://data.dtu.dk/articles/dataset/Raw_AI4Arctic_Sea_Ice_Challenge_Test_Dataset/21762848",
        "s3_prefix": "training_data/ai4arctic/raw_test/",
    },
}

DTU_API_BASE = "https://api.figshare.com/v2/articles"
PAGE_SIZE = 100


def get_article_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def s3_key_exists(bucket: str, key: str, s3: S3Client) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") == "404":
            return False
        raise


def get_file_links(article_url: str) -> list[dict]:
    article_id = get_article_id(article_url)
    all_files: list[dict] = []
    page = 1
    while True:
        resp = requests.get(
            f"{DTU_API_BASE}/{article_id}/files",
            params={"page": page, "page_size": PAGE_SIZE},
        )
        resp.raise_for_status()
        batch = resp.json()
        all_files.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        page += 1
    return all_files


def download_to_local(url: str, local_path: Path, filename: str) -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total_bytes = int(r.headers.get("content-length", 0))
                with tqdm(
                    total=total_bytes, unit="B", unit_scale=True, desc=f"↓ {filename}", leave=False
                ) as dl_pbar:
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            f.write(chunk)
                            dl_pbar.update(len(chunk))
            return
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            tqdm.write(
                f"  Download error on attempt {attempt}/{MAX_RETRIES}: {e}."
                f"Retrying in {wait:.0f}s..."
            )
            if local_path.exists():
                local_path.unlink()
            time.sleep(wait)


def main(bucket: str, profile: str | None = None) -> None:
    if profile:
        session = boto3.Session(profile_name=profile)
    else:
        session = boto3.Session()

    s3: S3Client = session.client("s3")

    dataset_files: dict[str, list[dict]] = {}
    for name, meta in DATASETS.items():
        files = get_file_links(meta["source_url"])
        dataset_files[name] = files
        print(f"{name}: {len(files)} files")

    all_files = [
        (name, file_meta, DATASETS[name]["s3_prefix"])
        for name, files in dataset_files.items()
        for file_meta in files
    ]

    download_dir = Path(tempfile.gettempdir())

    for name, file_meta, s3_prefix in tqdm(all_files, desc="Total files", unit="file"):
        filename = file_meta["name"]
        local_path = download_dir / filename
        s3_key = s3_prefix + filename

        if s3_key_exists(bucket, s3_key, s3):
            tqdm.write(f"Skipping {s3_key} (already in S3)")
            continue

        download_to_local(file_meta["download_url"], local_path, filename)

        with tqdm(
            total=local_path.stat().st_size,
            unit="B",
            unit_scale=True,
            desc=f"↑ {filename}",
            leave=False,
        ) as ul_pbar:
            s3.upload_file(
                str(local_path),
                bucket,
                s3_key,
                Callback=lambda n: ul_pbar.update(n),
            )

        local_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AI4Arctic datasets and upload to S3.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name to upload files to.")
    parser.add_argument("--profile", help="AWS CLI profile name to use for authentication.")
    args = parser.parse_args()
    bucket = args.bucket
    profile = args.profile

    main(bucket, profile)
