import argparse
import tempfile
import zipfile
from pathlib import Path

import boto3
import botocore.exceptions
from mypy_boto3_s3 import S3Client
from tqdm import tqdm

DATASETS = {
    "raw_train": {"input_prefix": "training_data/ai4arctic/raw_train/",
                  "output_prefix": "training_data/ai4arctic/raw_train/extracted/"},
}

def s3_key_exists(bucket: str, key: str, s3: S3Client) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") == "404":
            return False
        raise


def list_zip_keys(bucket: str, prefix: str, s3: S3Client) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj.get("Key", "").endswith(".zip"):
                keys.append(obj.get("Key", ""))
    return keys


def upload_file(local_path: Path, bucket: str, key: str, s3: S3Client) -> None:
    size = local_path.stat().st_size
    with tqdm(
        total=size, unit="B", unit_scale=True, desc=f"↑ {local_path.name}", leave=False
    ) as pbar:
        s3.upload_file(str(local_path), bucket, key, Callback=lambda n: pbar.update(n))


def process_zip(
    zip_s3_key: str,
    dataset_name: str,
    tmp_dir: Path,
    bucket: str,
    s3: S3Client,
) -> None:
    zip_name = Path(zip_s3_key).name
    local_zip = tmp_dir / zip_name

    size = s3.head_object(Bucket=bucket, Key=zip_s3_key)["ContentLength"]
    with tqdm(total=size, unit="B", unit_scale=True, desc=f"↓ {zip_name}", leave=False) as pbar:
        s3.download_file(
            bucket,
            zip_s3_key,
            str(local_zip),
            Callback=lambda n: pbar.update(n),
        )

    extract_dir = tmp_dir / zip_name.replace(".zip", "")
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(extract_dir)

    local_zip.unlink()

    extracted_files = [p for p in extract_dir.rglob("*") if p.is_file()]
    for extracted_path in tqdm(
        extracted_files, desc=f"Uploading {zip_name}", leave=False, unit="file"
    ):
        relative = extracted_path.relative_to(extract_dir)
        dest_key = f"{DATASETS[dataset_name]['output_prefix']}{relative}"

        if s3_key_exists(bucket, dest_key, s3):
            tqdm.write(f"  Skipping {dest_key} (already exists)")
            extracted_path.unlink()
            continue

        upload_file(extracted_path, bucket, dest_key, s3)
        extracted_path.unlink()


def main(bucket: str, profile: str | None = None) -> None:
    if profile:
        session = boto3.Session(profile_name=profile)
    else:        
        session = boto3.Session()
    
    s3: S3Client = session.client("s3")

    all_zips: list[tuple[str, str]] = []
    for name, info in DATASETS.items():
        keys = list_zip_keys(bucket, info["input_prefix"], s3)
        tqdm.write(f"{name}: {len(keys)} zip(s) found")
        all_zips.extend((name, key) for key in keys)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for dataset_name, zip_key in tqdm(all_zips, desc="Total zips", unit="zip"):
            tqdm.write(f"Processing {zip_key}")
            process_zip(zip_key, dataset_name, tmp_dir, bucket, s3)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AI4Arctic datasets and upload to S3.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name to upload files to.")
    parser.add_argument("--profile", help="AWS CLI profile name to use for authentication.")
    args = parser.parse_args()
    bucket = args.bucket
    profile = args.profile

    main(bucket, profile)
