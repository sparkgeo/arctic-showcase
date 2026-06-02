import boto3
import botocore.exceptions
import tempfile
import zipfile
from pathlib import Path
from tqdm import tqdm

s3 = boto3.client("s3")
BUCKET = "prescient-ice-data"

DATASETS = {
    "raw_train": "training_data/ai4arctic/raw_train/",
    "raw_test": "training_data/ai4arctic/raw_test/",
    "rtt_train": "training_data/ai4arctic/rtt_train/",
    "rtt_test": "training_data/ai4arctic/rtt_test/",
}

EXTRACTED_PREFIX_BASE = "training_data/ai4arctic/extracted/"


def s3_key_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def list_zip_keys(bucket: str, prefix: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".zip"):
                keys.append(obj["Key"])
    return keys


def upload_file(local_path: Path, bucket: str, key: str) -> None:
    size = local_path.stat().st_size
    with tqdm(total=size, unit="B", unit_scale=True, desc=f"↑ {local_path.name}", leave=False) as pbar:
        s3.upload_file(str(local_path), bucket, key, Callback=lambda n: pbar.update(n))


def process_zip(zip_s3_key: str, dataset_name: str, tmp_dir: Path) -> None:
    zip_name = Path(zip_s3_key).name
    local_zip = tmp_dir / zip_name

    size = s3.head_object(Bucket=BUCKET, Key=zip_s3_key)["ContentLength"]
    with tqdm(total=size, unit="B", unit_scale=True, desc=f"↓ {zip_name}", leave=False) as pbar:
        s3.download_file(
            BUCKET,
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
    for extracted_path in tqdm(extracted_files, desc=f"Uploading {zip_name}", leave=False, unit="file"):
        relative = extracted_path.relative_to(extract_dir)
        dest_key = f"{EXTRACTED_PREFIX_BASE}{dataset_name}/{relative}"

        if s3_key_exists(BUCKET, dest_key):
            tqdm.write(f"  Skipping {dest_key} (already exists)")
            extracted_path.unlink()
            continue

        upload_file(extracted_path, BUCKET, dest_key)
        extracted_path.unlink()


def main() -> None:
    all_zips: list[tuple[str, str]] = []
    for name, prefix in DATASETS.items():
        keys = list_zip_keys(BUCKET, prefix)
        tqdm.write(f"{name}: {len(keys)} zip(s) found")
        all_zips.extend((name, key) for key in keys)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for dataset_name, zip_key in tqdm(all_zips, desc="Total zips", unit="zip"):
            tqdm.write(f"Processing {zip_key}")
            process_zip(zip_key, dataset_name, tmp_dir)

    print("Done.")


if __name__ == "__main__":
    main()
