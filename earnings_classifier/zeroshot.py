"""Zero-shot FinBERT sentiment scores (no training, no selection).

The pinned FinBERT checkpoint ships a 3-way classifier (positive / negative /
neutral). Each text unit is scored as P(positive) - P(negative); an event's
score is the mean over its units. For documents the units are the same chunks
the embedding path encodes (`tokenize_document`); for headlines they are the
same up-to-`max_headlines` sequences (`tokenize_headlines`). Events without
units get NaN. Scores are rank-only (not calibrated probabilities).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from .config import Config
from .preprocessing import load_tokenizer, tokenize_document, tokenize_headlines
from .utils import get_logger

log = get_logger()


def load_sentiment_model(cfg: Config):
    from transformers import AutoModelForSequenceClassification

    from .utils import quiet_transformers

    quiet_transformers()
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name,
                                                               revision=cfg.model_revision)
    loaded = getattr(model.config, "_commit_hash", None)
    if loaded and loaded != cfg.model_revision:
        raise RuntimeError(f"{cfg.model_name}: loaded revision {loaded} != pinned "
                           f"{cfg.model_revision}")
    return model.eval()


def label_indices(id2label: Dict) -> Dict[str, int]:
    names = {str(v).lower(): int(k) for k, v in id2label.items()}
    if not {"positive", "negative"} <= set(names):
        raise ValueError(f"model labels {id2label} lack positive/negative; not a sentiment head")
    return {"positive": names["positive"], "negative": names["negative"]}


@torch.no_grad()
def _score_units(model, idx: Dict[str, int], input_ids, attention_mask, n_real: int,
                 device) -> float:
    if n_real == 0:
        return float("nan")
    logits = model(input_ids=input_ids[:n_real].to(device),
                   attention_mask=attention_mask[:n_real].to(device)).logits
    p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return float(np.mean(p[:, idx["positive"]] - p[:, idx["negative"]]))


def sentiment_scores(events: Sequence, cfg: Config, device, stream: str = "document",
                     model=None, tokenizer=None) -> np.ndarray:
    """Mean P(positive) - P(negative) per event for `stream` in {"document", "headlines"}."""
    if stream not in ("document", "headlines"):
        raise ValueError(f"unknown stream {stream!r}")
    tokenizer = tokenizer or load_tokenizer(cfg)
    model = (model or load_sentiment_model(cfg)).to(device)
    idx = label_indices(model.config.id2label)
    out: List[float] = []
    for i, e in enumerate(events):
        t = tokenize_document(e.document, tokenizer, cfg) if stream == "document" \
            else tokenize_headlines(e.headlines, tokenizer, cfg)
        out.append(_score_units(model, idx, t["input_ids"], t["attention_mask"],
                                int(t["n_real"]), device))
        if (i + 1) % 100 == 0:
            log.info("  zero-shot %s sentiment %d/%d", stream, i + 1, len(events))
    return np.array(out, dtype=float)
