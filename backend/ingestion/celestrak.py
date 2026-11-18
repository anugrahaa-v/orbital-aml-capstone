"""CelesTrak GP client (the single dataset used by ORBITALX AI).

- Endpoint: https://celestrak.org/NORAD/elements/gp.php?GROUP=<g>&FORMAT=json
- No API key. Fair use: one download per update cycle (~2 h); we enforce a
  cache-first policy so repeated runs never re-download inside that window.
- Nothing is ever fabricated: on network failure we raise, or fall back to the
  most recent *real* cached download and say so (from_cache=True in registry).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import requests

from .registry import SourceRecord, append, sha256_bytes, utc_now_iso

BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"
ACCESS = "Public, no key; CelesTrak fair-use policy (one download per ~2 h update cycle)."
ATTRIBUTION = "CelesTrak (celestrak.org), GP data derived from 18th/19th SDS via Space-Track."


class CelesTrakError(RuntimeError):
    pass


def _latest_cache(cache_dir: Path, group: str) -> Path | None:
    files = sorted(cache_dir.glob(f"gp_{group}_*.json"))
    return files[-1] if files else None


def fetch_group(group: str, data_dir: Path = Path("data"), min_interval_s: int = 7200,
                allow_stale_cache: bool = True, timeout_s: int = 30,
                session: requests.Session | None = None) -> tuple[list[dict], SourceRecord]:
    cache_dir = data_dir / "cache"
    registry = data_dir / "registry" / "source_registry.jsonl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}?GROUP={group}&FORMAT=json"

    cached = _latest_cache(cache_dir, group)
    if cached is not None and (time.time() - cached.stat().st_mtime) < min_interval_s:
        raw = cached.read_bytes()
        return _finish(raw, url, cached, True, registry)

    sess = session or requests.Session()
    try:
        resp = sess.get(url, timeout=timeout_s, headers={"User-Agent": "ORBITALX-AI/0.1 (MSc research)"})
        resp.raise_for_status()
        raw = resp.content
        _ = json.loads(raw)  # fail early if not JSON (CelesTrak returns plain text on errors)
    except (requests.RequestException, ValueError) as exc:
        if allow_stale_cache and cached is not None:
            return _finish(cached.read_bytes(), url, cached, True, registry)
        raise CelesTrakError(f"Could not retrieve {url}: {exc}") from exc

    path = cache_dir / f"gp_{group}_{utc_now_iso().replace(':', '')}.json"
    path.write_bytes(raw)
    return _finish(raw, url, path, False, registry)


def _finish(raw: bytes, url: str, path: Path, from_cache: bool, registry: Path):
    records = json.loads(raw)
    if not isinstance(records, list):
        raise CelesTrakError(f"Unexpected payload from {url}: {str(records)[:200]}")
    rec = SourceRecord("CelesTrak GP (OMM JSON)", url, utc_now_iso(), sha256_bytes(raw),
                       len(records), ACCESS, ATTRIBUTION, str(path), from_cache)
    append(rec, registry)
    return records, rec
