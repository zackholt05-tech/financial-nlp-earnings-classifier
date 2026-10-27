"""Precompute frozen FinBERT embeddings for every event.

For the default (frozen-encoder) path we run FinBERT once per event, pool its
document chunks and news headlines into two fixed vectors, and cache the
result. Head training and the logistic-regression baseline then operate on these
vectors: fast, CPU/MPS-friendly, and reproducible.

The cache key covers event identities, text content, labels, splits, the pinned
model/tokenizer revision and every tokenization/pooling setting
(cache.embeddings_cache_key). Labels and splits returned to callers are always
taken from the live events, never from the cache file.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .cache import (
    embeddings_cache_key,
    embeddings_cache_path,
    load_embeddings_cache,
    save_embeddings_cache,
)
from .config import Config
from .data_collection import EarningsEvent
from .model import FinBERTEncoder
from .preprocessing import (
    EmbeddingDataset,
    load_tokenizer,
    tokenize_headlines,
    tokenize_document,
)
from .utils import get_logger

log = get_logger()

_VECTOR_KEYS = ("t_emb", "n_emb", "t_present", "n_present")


@torch.no_grad()
def compute_embeddings(
    events: List[EarningsEvent],
    cfg: Config,
    device: torch.device,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> Dict[str, np.ndarray]:
    """Encode all events into fixed vectors. Returns a dict of numpy arrays plus
    `event_ids`, `labels`, `splits`, `cache_key` and `cache_hit`."""
    key, _ = embeddings_cache_key(events, cfg, device.type)
    event_ids = [e.event_id for e in events]
    path = embeddings_cache_path(cache_dir, key) if cache_dir else None

    vectors = load_embeddings_cache(path, key, event_ids) if (path and use_cache) else None
    cache_hit = vectors is not None
    if cache_hit:
        log.info("Loaded cached embeddings from %s", path)
    else:
        vectors = _encode(events, cfg, device)
        if path and write_cache:
            save_embeddings_cache(path, key, event_ids, vectors)
            log.info("Cached embeddings to %s", path)

    return dict(
        vectors,
        event_ids=np.array(event_ids),
        labels=np.array([e.label for e in events], dtype=np.int64),
        splits=np.array([e.split for e in events]),
        cache_key=key,
        cache_hit=cache_hit,
        encode_device=device.type,
        cache_path=path if (path and (use_cache or write_cache)) else "",
    )


def _encode(events: List[EarningsEvent], cfg: Config, device: torch.device) -> Dict[str, np.ndarray]:
    tokenizer = load_tokenizer(cfg)
    encoder = FinBERTEncoder(cfg).to(device).eval()
    H = encoder.hidden_size

    N = len(events)
    out = {
        "t_emb": np.zeros((N, H), dtype=np.float32),
        "n_emb": np.zeros((N, H), dtype=np.float32),
        "t_present": np.zeros(N, dtype=np.float32),
        "n_present": np.zeros(N, dtype=np.float32),
    }
    log.info("Encoding %d events with %s@%s ...", N, cfg.model_name, cfg.model_revision[:7])
    for i, e in enumerate(events):
        t = tokenize_document(e.document, tokenizer, cfg)
        n = tokenize_headlines(e.headlines, tokenizer, cfg)
        tv, tp = encoder.encode_bag(
            t["input_ids"].unsqueeze(0).to(device),
            t["attention_mask"].unsqueeze(0).to(device),
            t["n_real"].unsqueeze(0).to(device),
        )
        nv, npres = encoder.encode_bag(
            n["input_ids"].unsqueeze(0).to(device),
            n["attention_mask"].unsqueeze(0).to(device),
            n["n_real"].unsqueeze(0).to(device),
        )
        out["t_emb"][i] = tv.squeeze(0).float().cpu().numpy()
        out["n_emb"][i] = nv.squeeze(0).float().cpu().numpy()
        out["t_present"][i] = float(tp.item())
        out["n_present"][i] = float(npres.item())
        if (i + 1) % 50 == 0:
            log.info("  encoded %d/%d", i + 1, N)
    return out


def _mask(arr: Dict[str, np.ndarray], split: str) -> np.ndarray:
    return arr["splits"] == split


def make_embedding_dataset(arr: Dict[str, np.ndarray], split: str,
                           modalities: str = "both") -> EmbeddingDataset:
    """`modalities` in {"both", "document", "headlines"} masks the other stream
    (zero vector + presence flag 0) for ablations."""
    m = _mask(arr, split)
    t_emb, n_emb = arr["t_emb"][m].copy(), arr["n_emb"][m].copy()
    t_pres, n_pres = arr["t_present"][m].copy(), arr["n_present"][m].copy()
    if modalities == "document":
        n_emb[:], n_pres[:] = 0.0, 0.0
    elif modalities == "headlines":
        t_emb[:], t_pres[:] = 0.0, 0.0
    elif modalities != "both":
        raise ValueError(f"unknown modalities {modalities!r}")
    return EmbeddingDataset(
        torch.from_numpy(t_emb), torch.from_numpy(n_emb),
        torch.from_numpy(t_pres), torch.from_numpy(n_pres),
        torch.from_numpy(arr["labels"][m].astype(np.float32)),
    )


def split_concat_features(arr: Dict[str, np.ndarray], split: str, modalities: str = "both"):
    """Concatenated [t_emb | n_emb | presence flags] features + labels for the
    LR baseline (the same information the fusion head receives)."""
    ds = make_embedding_dataset(arr, split, modalities)
    t, n = ds.t_emb.numpy(), ds.n_emb.numpy()
    tp, npres = ds.t_present.numpy(), ds.n_present.numpy()
    X = np.concatenate([t * tp[:, None], n * npres[:, None], tp[:, None], npres[:, None]], axis=1)
    return X, arr["labels"][_mask(arr, split)]
