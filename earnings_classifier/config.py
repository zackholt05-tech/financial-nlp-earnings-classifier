"""Central configuration for the post-earnings direction classifier.

Everything tunable lives here so the pipeline (data -> embeddings -> model ->
training -> evaluation) reads from a single, serializable source of truth.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List


# A small, liquid, well-covered universe. Override via Config(tickers=[...]).
DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "V", "JNJ", "WMT", "PG", "XOM", "HD", "BAC", "KO", "PEP", "CSCO",
    "INTC", "NFLX", "AMD", "CRM", "ADBE", "ORCL", "DIS", "NKE", "MCD",
    "COST", "TMO", "ABBV",
]

# Explicit experiment modes. Synthetic modes are pipeline checks, never market evidence.
REAL = "real"
NEGATIVE_CONTROL = "synthetic-negative-control"   # labels independent of text
POSITIVE_CONTROL = "synthetic-positive-control"   # text deliberately encodes the label
DATA_MODES = (REAL, NEGATIVE_CONTROL, POSITIVE_CONTROL)

# Text providers for real mode (see providers.py / docs/DATA_SOURCES.md).
DOCUMENT_PROVIDERS = ("sec-edgar", "fmp", "none")   # earnings press release | call transcript
HEADLINE_PROVIDERS = ("finnhub", "none")

# Pinned Hugging Face commits (verified 2026-10-27; see docs/DATA_SOURCES.md).
FINBERT = "ProsusAI/finbert"
FINBERT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
TINY_BERT = "google/bert_uncased_L-2_H-128_A-2"
TINY_BERT_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# Fields from pre-audit configs that no longer exist (handled by `from_dict`).
_LEGACY_FIELDS = ("synthetic", "synthetic_signal", "artifacts_dir")
# Fields renamed when the text stream was generalized from transcripts to documents.
_RENAMED_FIELDS = {"transcript_chunk_len": "document_chunk_len",
                   "max_transcript_chunks": "max_document_chunks",
                   "transcript_match_days": "document_match_days"}


@dataclass
class Config:
    # ---- Model / encoder ----
    model_name: str = FINBERT               # FinBERT base encoder (768-d)
    model_revision: str = FINBERT_REVISION  # full commit hash; also used for the tokenizer
    pooling: str = "mean"                   # "mean" (mask-aware) or "cls"
    freeze_encoder: bool = True             # frozen -> precompute embeddings + train head
    # Fusion of the two 768-d streams.
    use_interactions: bool = True           # also include |t-n| and t*n
    use_presence_flags: bool = True         # tell the head when a modality was absent
    head_hidden: int = 64                   # width when head_layers >= 1
    head_dropout: float = 0.5               # heavy dropout to fight memorization
    head_layers: int = 0                    # 0 == dropout + one linear layer (a logistic
                                            # head on the fused features; the default)

    # ---- Text windowing ----
    document_chunk_len: int = 384           # tokens per document chunk (<=512)
    max_document_chunks: int = 8            # cap chunks/event for compute
    headline_max_len: int = 48              # tokens per headline
    max_headlines: int = 16                 # cap headlines/event

    # ---- Labeling / information timeline (see docs/PLAN.md) ----
    horizon_days: int = 3                   # trading sessions from entry open to exit close
    label_threshold: float = 0.0            # return > threshold -> "up" (1)
    deadband: float = 0.0                   # drop |return| <= deadband as ambiguous
    news_window_days: int = 3               # headline lookback window before the announcement
    document_match_days: int = 3            # document date must be in [A-1, A+this] days

    # ---- Temporal split (NEVER random for this task) ----
    val_fraction: float = 0.15              # most-recent-but-one slice
    test_fraction: float = 0.20             # most-recent slice (held out)
    val_start_date: str = ""                # optional explicit boundaries (YYYY-MM-DD);
    test_start_date: str = ""               # when set they override the fractions

    # ---- Optimization ----
    batch_size: int = 32
    head_lr: float = 1e-3
    encoder_lr: float = 2e-5                # only used when freeze_encoder is False
    weight_decay: float = 0.05
    epochs: int = 40
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    early_stop_patience: int = 8            # epochs without val-AUC improvement
    use_pos_weight: bool = True             # rebalance BCE for class imbalance

    # ---- Data source ----
    data_mode: str = REAL                   # one of DATA_MODES
    n_synthetic_events: int = 400           # events to fabricate in synthetic modes
    min_real_events: int = 60               # real mode fails below this many usable events
    tickers: List[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    earnings_per_ticker: int = 40           # most recent announcements to request from yfinance
    document_provider: str = "sec-edgar"    # one of DOCUMENT_PROVIDERS
    headline_provider: str = "finnhub"      # one of HEADLINE_PROVIDERS
    min_document_chars: int = 200           # shorter documents count as missing (stub exhibits)

    # ---- Runtime ----
    seed: int = 42
    device: str = "auto"                    # "auto" | "cpu" | "mps" | "cuda"
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    runs_dir: str = "runs"
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.data_mode not in DATA_MODES:
            raise ValueError(f"data_mode must be one of {DATA_MODES}, got {self.data_mode!r}")
        if self.document_provider not in DOCUMENT_PROVIDERS:
            raise ValueError(f"document_provider must be one of {DOCUMENT_PROVIDERS}")
        if self.headline_provider not in HEADLINE_PROVIDERS:
            raise ValueError(f"headline_provider must be one of {HEADLINE_PROVIDERS}")
        if not _COMMIT_RE.match(self.model_revision):
            raise ValueError(
                f"model_revision must be a full 40-character commit hash so runs are "
                f"reproducible; got {self.model_revision!r} for {self.model_name}"
            )

    # ---- Derived helpers -------------------------------------------------
    @property
    def is_synthetic(self) -> bool:
        return self.data_mode != REAL

    @property
    def synthetic_signal(self) -> bool:
        return self.data_mode == POSITIVE_CONTROL

    @property
    def fused_dim_multiplier(self) -> int:
        """How many embedding-sized blocks the fusion produces (before presence flags)."""
        return 4 if self.use_interactions else 2

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "Config":
        """Build a Config, translating pre-audit fields (synthetic/synthetic_signal,
        transcript_* names)."""
        d = dict(d)
        if "data_mode" not in d and ("synthetic" in d or "synthetic_signal" in d):
            if d.get("synthetic_signal"):
                d["data_mode"] = POSITIVE_CONTROL
            elif d.get("synthetic"):
                d["data_mode"] = NEGATIVE_CONTROL
            else:
                d["data_mode"] = REAL
        for k in _LEGACY_FIELDS:
            d.pop(k, None)
        for old, new in _RENAMED_FIELDS.items():
            if old in d:
                d.setdefault(new, d.pop(old))
        if "model_revision" not in d:
            known = {FINBERT: FINBERT_REVISION, TINY_BERT: TINY_BERT_REVISION}
            if d.get("model_name", FINBERT) not in known:
                raise ValueError("legacy config without model_revision for an unknown model")
            d["model_revision"] = known[d.get("model_name", FINBERT)]
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def tiny(cls, **overrides) -> "Config":
        """Fast preset for smoke tests and the demo: a tiny BERT + tiny everything."""
        defaults = dict(
            # Official Google BERT-tiny: hidden=128, standard bert-base-uncased
            # vocab (same family as FinBERT), ships a working fast tokenizer.
            model_name=TINY_BERT,
            model_revision=TINY_BERT_REVISION,
            document_chunk_len=64,
            max_document_chunks=3,
            headline_max_len=24,
            max_headlines=4,
            head_hidden=16,
            batch_size=8,
            epochs=5,
            n_synthetic_events=200,  # large enough that val/test are never single-class
            data_mode=NEGATIVE_CONTROL,
        )
        defaults.update(overrides)
        return cls(**defaults)
