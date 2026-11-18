"""Data-source registry: append-only JSONL log of every retrieval (provenance)."""
from __future__ import annotations
import json, hashlib, datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class SourceRecord:
    source: str
    query_url: str
    retrieved_utc: str
    dataset_version: str      # sha256 of the raw payload
    n_records_raw: int
    access_conditions: str
    attribution: str
    raw_file: str
    from_cache: bool


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append(record: SourceRecord, registry_path: Path) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_all(registry_path: Path) -> list[dict]:
    if not registry_path.exists():
        return []
    with registry_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
