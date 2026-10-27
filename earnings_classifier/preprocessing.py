"""Tokenization, windowing, and torch datasets.

Earnings documents are long (the median SEC press release with its tables is
~29k characters, very roughly 7k tokens; call transcripts are longer), so each is split
into consecutive, NON-overlapping chunks of `document_chunk_len` tokens
(default 384), keeping at most `max_document_chunks` (default 8, i.e. ~3k
tokens: the opening results/narrative of a press release, or the prepared
remarks of a call transcript). Headlines are short and handled as a small
bag of sequences (the first `max_headlines`, stored most-recent first). Two consumption paths are supported:

  * EmbeddingDataset - fixed precomputed (document_emb, news_emb) vectors for
    fast head-only training (the default, frozen-encoder path).
  * RawEventDataset  - per-event token tensors + `collate_raw` for end-to-end
    fine-tuning of the encoder.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .config import Config
from .data_collection import EarningsEvent


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    from .utils import quiet_transformers

    quiet_transformers()
    return AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision)


def tokenize_document(text: str, tokenizer, cfg: Config) -> Dict[str, torch.Tensor]:
    """Split a long document into <=max_document_chunks padded windows.

    Returns dict with input_ids/attention_mask of shape [n_chunks, chunk_len]
    and `n_real` = number of non-padding chunks (>=1; 0 signals empty text).
    """
    L = cfg.document_chunk_len
    body = L - 2  # room for [CLS]/[SEP]
    cls, sep, pad = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id

    ids = tokenizer.encode(text or "", add_special_tokens=False)
    chunks: List[List[int]] = []
    if ids:
        for start in range(0, len(ids), body):
            window = ids[start : start + body]
            chunks.append([cls] + window + [sep])
            if len(chunks) >= cfg.max_document_chunks:
                break
    n_real = len(chunks)
    if not chunks:  # empty document -> single all-pad chunk, n_real=0
        chunks = [[pad] * L]

    input_ids, attn = [], []
    for ch in chunks:
        mask = [1] * len(ch) + [0] * (L - len(ch))
        ch = ch + [pad] * (L - len(ch))
        input_ids.append(ch[:L])
        attn.append(mask[:L])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "n_real": torch.tensor(n_real, dtype=torch.long),
    }


def tokenize_headlines(headlines: List[str], tokenizer, cfg: Config) -> Dict[str, torch.Tensor]:
    """Tokenize up to max_headlines into [n_headlines, headline_max_len]."""
    heads = [h for h in (headlines or []) if h.strip()][: cfg.max_headlines]
    n_real = len(heads)
    if not heads:
        heads = [""]  # placeholder row, n_real=0

    enc = tokenizer(
        heads,
        padding="max_length",
        truncation=True,
        max_length=cfg.headline_max_len,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "n_real": torch.tensor(n_real, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Frozen path: fixed precomputed embeddings
# ---------------------------------------------------------------------------
class EmbeddingDataset(Dataset):
    """Holds fixed (document_emb, news_emb) vectors + presence flags + labels."""

    def __init__(
        self,
        t_emb: torch.Tensor,
        n_emb: torch.Tensor,
        t_present: torch.Tensor,
        n_present: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.t_emb = t_emb
        self.n_emb = n_emb
        self.t_present = t_present
        self.n_present = n_present
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "t_emb": self.t_emb[i],
            "n_emb": self.n_emb[i],
            "t_present": self.t_present[i],
            "n_present": self.n_present[i],
            "label": self.labels[i],
        }


# ---------------------------------------------------------------------------
# End-to-end path: raw tokens with a padding collate over the chunk/headline dim
# ---------------------------------------------------------------------------
class RawEventDataset(Dataset):
    def __init__(self, events: List[EarningsEvent], tokenizer, cfg: Config):
        self.events = events
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, i: int) -> Dict:
        e = self.events[i]
        t = tokenize_document(e.document, self.tokenizer, self.cfg)
        n = tokenize_headlines(e.headlines, self.tokenizer, self.cfg)
        return {"t": t, "n": n, "label": torch.tensor(float(e.label))}


def _pad_stack(seqs: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """Pad a list of [k_i, L] tensors to [B, max_k, L] and stack."""
    max_k = max(s.shape[0] for s in seqs)
    L = seqs[0].shape[1]
    out = torch.full((len(seqs), max_k, L), pad_value, dtype=seqs[0].dtype)
    for b, s in enumerate(seqs):
        out[b, : s.shape[0]] = s
    return out


def collate_raw(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate variable chunk/headline counts into padded batch tensors."""
    out: Dict[str, torch.Tensor] = {}
    for stream in ("t", "n"):
        out[f"{stream}_input_ids"] = _pad_stack([b[stream]["input_ids"] for b in batch])
        out[f"{stream}_attention_mask"] = _pad_stack(
            [b[stream]["attention_mask"] for b in batch]
        )
        out[f"{stream}_n_real"] = torch.stack([b[stream]["n_real"] for b in batch])
    out["label"] = torch.stack([b["label"] for b in batch])
    return out
