"""Provenance-keyed caches for events and frozen-encoder embeddings.

A cache entry is only ever returned to a request whose key material is
identical to the material it was written with. Keys are SHA-256 hashes of
canonical JSON; the full material is stored inside the file and re-verified on
load, so a file at the right path with the wrong contents is an error, not a hit.
Legacy (pre-provenance) cache files are never read.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import REAL, Config

EVENTS_FORMAT = "earnings-events"
EMBEDDINGS_FORMAT = "earnings-embeddings"
# Bump when the event-generation logic changes in a way the config cannot express.
COLLECTOR_VERSION = 6
# Bump when the embedding computation (tokenization/pooling code) changes.
EMBEDDER_VERSION = 2


class CacheError(RuntimeError):
    """A cache file exists at the expected path but does not match its key."""


def stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
def events_cache_key(cfg: Config, providers=None) -> Tuple[str, Dict]:
    from .data_collection import SCHEMA_VERSION

    material: Dict = {
        "format": EVENTS_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "collector_version": COLLECTOR_VERSION,
        "data_mode": cfg.data_mode,
        "tickers": list(cfg.tickers),
        "horizon_days": cfg.horizon_days,
        "label_threshold": cfg.label_threshold,
        "deadband": cfg.deadband,
        "news_window_days": cfg.news_window_days,
    }
    if cfg.data_mode == REAL:
        material.update(
            earnings_per_ticker=cfg.earnings_per_ticker,
            document_match_days=cfg.document_match_days,
            min_document_chars=cfg.min_document_chars,
            text_providers={
                "documents": getattr(getattr(providers, "documents", None), "source", None),
                "document_type": getattr(getattr(providers, "documents", None),
                                         "document_type", None),
                "headlines": getattr(getattr(providers, "news", None), "source", None),
            },
        )
    else:
        material.update(n_synthetic_events=cfg.n_synthetic_events, seed=cfg.seed,
                        max_headlines=cfg.max_headlines)
    return stable_hash(material), material


def save_events_cache(path: str, events, key: str, material: Dict, report: Dict) -> None:
    from .data_collection import dataset_fingerprint

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "format": EVENTS_FORMAT,
        "cache_key": key,
        "key_material": material,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fingerprint(events),
        "report": report,
        "events": [e.content_dict() for e in events],
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def load_events_cache(path: str, key: str, material: Dict):
    """Return (events, meta) on a verified hit, None if no file; raise on mismatch."""
    from .data_collection import EarningsEvent, dataset_fingerprint

    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or payload.get("format") != EVENTS_FORMAT:
        raise CacheError(f"{path} is not a provenance-tagged events cache (legacy format?). "
                         "Delete it or run with --refresh-cache.")
    if payload.get("cache_key") != key or payload.get("key_material") != material:
        raise CacheError(f"{path}: stored key/material does not match the request")
    events = [EarningsEvent(**d) for d in payload["events"]]
    if dataset_fingerprint(events) != payload.get("dataset_fingerprint"):
        raise CacheError(f"{path}: events do not match the stored dataset fingerprint")
    return events, payload


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def embeddings_cache_key(events: Sequence, cfg: Config, device_type: str) -> Tuple[str, Dict]:
    """`device_type` is part of the key: CPU and MPS/CUDA kernels give slightly
    different floats, so embeddings are only reused on the device that made them."""
    from .data_collection import text_hash

    material = {
        "format": EMBEDDINGS_FORMAT,
        "embedder_version": EMBEDDER_VERSION,
        "encode_device": device_type,
        "model_name": cfg.model_name,
        "model_revision": cfg.model_revision,
        "tokenizer": cfg.model_name,
        "tokenizer_revision": cfg.model_revision,
        "pooling": cfg.pooling,
        "document_chunk_len": cfg.document_chunk_len,
        "max_document_chunks": cfg.max_document_chunks,
        "headline_max_len": cfg.headline_max_len,
        "max_headlines": cfg.max_headlines,
        "events": [
            [e.event_id, text_hash(e.document), text_hash(json.dumps(e.headlines)),
             int(e.label), e.split]
            for e in events
        ],
    }
    return stable_hash(material), material


def embeddings_cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"embeddings-{key[:16]}.npz")


def save_embeddings_cache(path: str, key: str, event_ids: List[str], arrays: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, cache_key=np.array(key), event_ids=np.array(event_ids), **arrays)
    os.replace(tmp, path)


def load_embeddings_cache(path: str, key: str, event_ids: List[str]) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        if "cache_key" not in z.files or str(z["cache_key"]) != key:
            raise CacheError(f"{path}: stored key does not match the request")
        if z["event_ids"].tolist() != list(event_ids):
            raise CacheError(f"{path}: event order does not match the request")
        return {k: z[k] for k in ("t_emb", "n_emb", "t_present", "n_present")}
