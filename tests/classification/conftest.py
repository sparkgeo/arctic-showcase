import io
from collections.abc import Iterator
from dataclasses import dataclass, field

import pandas as pd
import pytest


@dataclass
class FakePaginator:
    objects: dict[str, bytes]

    def paginate(
        self, Bucket: str, Prefix: str, Delimiter: str | None = None
    ) -> Iterator[dict[str, list[dict[str, str]]]]:
        keys = [k for k in self.objects if k.startswith(Prefix)]
        if Delimiter is None:
            yield {"Contents": [{"Key": k} for k in sorted(keys)]}
            return

        common_prefixes: set[str] = set()
        contents: list[dict[str, str]] = []
        for key in keys:
            rest = key[len(Prefix) :]
            if Delimiter in rest:
                common_prefixes.add(Prefix + rest.split(Delimiter, 1)[0] + Delimiter)
            else:
                contents.append({"Key": key})
        yield {
            "CommonPrefixes": [{"Prefix": p} for p in sorted(common_prefixes)],
            "Contents": contents,
        }


@dataclass
class FakeS3:
    """Minimal in-memory stand-in for the S3Client surface tables.py uses --
    list_objects_v2 pagination (with and without a Delimiter) and get_object."""

    objects: dict[str, bytes] = field(default_factory=dict)

    def put_dataframe(self, key: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf)
        self.objects[key] = buf.getvalue()

    def get_paginator(self, operation_name: str) -> FakePaginator:
        assert operation_name == "list_objects_v2"
        return FakePaginator(self.objects)

    def get_object(self, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
        return {"Body": io.BytesIO(self.objects[Key])}


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()
