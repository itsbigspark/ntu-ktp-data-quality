"""
Source connectors — pull a DataFrame from any supported data source.

A source is anything that can produce a table: a local file, an object in S3,
a JSON HTTP API, or the Companies House API. Every connector implements a single
method, ``fetch() -> pandas.DataFrame``, so the batch runner (and, later, the
autonomous pipeline and the agent) can treat every source identically.

Parse a source from a string spec::

    from dataqualify.sources import parse_source

    parse_source("customers.csv")                      # local file
    parse_source("s3://my-bucket/incoming/data.csv")   # object storage
    parse_source("https://api.example.com/records")    # JSON HTTP API
    parse_source("companies-house:12345678,SC123456")  # Companies House API

Heavy/cloud dependencies are optional: S3Source needs ``boto3`` (the ``[aws]``
extra); the HTTP connectors use only the standard library.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class SourceConnector(ABC):
    """Base class for all data sources. One method: ``fetch``."""

    #: short human-readable identifier recorded on the batch (e.g. the URI).
    name: str = "source"

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Return the source data as a pandas DataFrame."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.name!r})"


def _read_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Parse raw bytes into a DataFrame based on the filename extension."""
    lower = filename.lower()
    buf = io.BytesIO(data)
    if lower.endswith((".parquet", ".pq")):
        return pd.read_parquet(buf)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    if lower.endswith(".json"):
        return pd.json_normalize(json.loads(data.decode("utf-8")))
    return pd.read_csv(buf)


class FileSource(SourceConnector):
    """A local file: CSV, Parquet, Excel, or JSON."""

    def __init__(self, path: str):
        self.path = path
        self.name = f"file://{os.path.abspath(path)}"

    def fetch(self) -> pd.DataFrame:
        with open(self.path, "rb") as fh:
            return _read_bytes(fh.read(), self.path)


class S3Source(SourceConnector):
    """An object in S3: ``s3://bucket/key`` (CSV/Parquet/Excel/JSON)."""

    def __init__(self, uri: str, region: Optional[str] = None):
        if not uri.startswith("s3://"):
            raise ValueError(f"not an s3 uri: {uri}")
        self.uri = uri
        self.bucket, _, self.key = uri[len("s3://"):].partition("/")
        self.region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.name = uri

    def fetch(self) -> pd.DataFrame:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "S3 sources require boto3. Install with: pip install 'dataqualify[aws]'"
            ) from exc
        s3 = boto3.client("s3", region_name=self.region)
        obj = s3.get_object(Bucket=self.bucket, Key=self.key)
        return _read_bytes(obj["Body"].read(), self.key)


class HttpApiSource(SourceConnector):
    """A JSON HTTP API. Fetches the URL, locates a list of records, and
    normalises it into a table.

    Args:
        url: the endpoint to GET.
        record_path: dotted path to the list of records inside the JSON
            (e.g. ``"items"`` or ``"data.results"``). If omitted, the top-level
            value is used when it is a list, otherwise the whole object becomes
            a single row.
        headers: optional request headers (auth, etc.).
    """

    def __init__(self, url: str, record_path: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None, timeout: int = 30):
        self.url = url
        self.record_path = record_path
        self.headers = headers or {}
        self.timeout = timeout
        self.name = url

    def _get_json(self) -> object:
        req = urllib.request.Request(self.url, headers={"Accept": "application/json", **self.headers})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def fetch(self) -> pd.DataFrame:
        payload = self._get_json()
        records = payload
        if self.record_path:
            for part in self.record_path.split("."):
                records = records[part]
        if isinstance(records, dict):
            records = [records]
        return pd.json_normalize(records)


class CompaniesHouseSource(SourceConnector):
    """The UK Companies House API — fetch one or more companies by number.

    Requires an API key in the ``CH_API_KEY`` environment variable (obtain one
    from developer.company-information.service.gov.uk). Returns one row per
    company with the profile fields flattened.
    """

    BASE = "https://api.company-information.service.gov.uk/company/"

    def __init__(self, numbers: List[str], api_key: Optional[str] = None):
        self.numbers = [n.strip() for n in numbers if n.strip()]
        self.api_key = api_key or os.environ.get("CH_API_KEY", "")
        self.name = f"companies-house:{','.join(self.numbers)}"

    def fetch(self) -> pd.DataFrame:
        if not self.api_key:
            raise RuntimeError(
                "Companies House requires an API key in the CH_API_KEY environment variable."
            )
        import base64
        token = base64.b64encode(f"{self.api_key}:".encode()).decode()
        rows = []
        for number in self.numbers:
            src = HttpApiSource(self.BASE + number,
                                headers={"Authorization": f"Basic {token}"})
            payload = src._get_json()
            if isinstance(payload, dict):
                rows.append(payload)
        return pd.json_normalize(rows)


def parse_source(spec: str, **kwargs) -> SourceConnector:
    """Turn a string spec into the appropriate SourceConnector."""
    spec = spec.strip()
    if spec.startswith("s3://"):
        return S3Source(spec, region=kwargs.get("region"))
    if spec.startswith(("http://", "https://")):
        return HttpApiSource(spec, record_path=kwargs.get("record_path"),
                             headers=kwargs.get("headers"))
    if spec.startswith("companies-house:"):
        numbers = spec[len("companies-house:"):].split(",")
        return CompaniesHouseSource(numbers, api_key=kwargs.get("api_key"))
    return FileSource(spec)
