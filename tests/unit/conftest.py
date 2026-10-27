"""Shared fixtures for network-free unit tests.

`FakeTokenizer` / `FakeEncoder` stand in for the Hugging Face tokenizer and the
FinBERT encoder so cache, provenance and pipeline tests never download a model.
The fake encoder is deterministic and content-sensitive (different text gives
different vectors), which is what the cache tests need.
"""

from __future__ import annotations

import zlib
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

from earnings_classifier.model import masked_mean_bag, masked_mean_tokens

VOCAB = 997
HIDDEN = 8


class FakeTokenizer:
    cls_token_id, sep_token_id, pad_token_id = 1, 2, 0

    def _ids(self, text: str) -> List[int]:
        return [3 + zlib.crc32(w.encode()) % (VOCAB - 3) for w in (text or "").split()]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = self._ids(text)
        return [self.cls_token_id] + ids + [self.sep_token_id] if add_special_tokens else ids

    def __call__(self, texts, padding="max_length", truncation=True, max_length=16,
                 return_tensors="pt") -> Dict[str, torch.Tensor]:
        rows, masks = [], []
        for t in texts:
            ids = ([self.cls_token_id] + self._ids(t) + [self.sep_token_id])[:max_length]
            if not t:
                ids = []
            masks.append([1] * len(ids) + [0] * (max_length - len(ids)))
            rows.append(ids + [self.pad_token_id] * (max_length - len(ids)))
        return {"input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long)}


class FakeEncoder(nn.Module):
    instances = 0
    bag_calls = 0

    def __init__(self, cfg=None):
        super().__init__()
        FakeEncoder.instances += 1
        g = torch.Generator().manual_seed(0)
        self.table = nn.Parameter(torch.randn(VOCAB, HIDDEN, generator=g), requires_grad=False)
        self.hidden_size = HIDDEN
        self.cfg = cfg

    def encode_bag(self, input_ids, attention_mask, n_real):
        FakeEncoder.bag_calls += 1
        B, K, L = input_ids.shape
        tok = self.table[input_ids.reshape(B * K, L)]
        seq = masked_mean_tokens(tok, attention_mask.reshape(B * K, L)).reshape(B, K, HIDDEN)
        return masked_mean_bag(seq, n_real)


@pytest.fixture
def fake_models(monkeypatch):
    """Replace the real tokenizer/encoder used by embeddings.compute_embeddings."""
    import earnings_classifier.embeddings as emb_mod

    FakeEncoder.instances = 0
    FakeEncoder.bag_calls = 0
    monkeypatch.setattr(emb_mod, "FinBERTEncoder", FakeEncoder)
    monkeypatch.setattr(emb_mod, "load_tokenizer", lambda cfg: FakeTokenizer())
    return FakeEncoder


# ---------------------------------------------------------------------------
# Hand-made real-mode fixtures (fake ticker ACME; no provider data is redistributed)
# ---------------------------------------------------------------------------
from datetime import date, datetime, timedelta, timezone  # noqa: E402

import pandas as pd  # noqa: E402

from earnings_classifier.config import REAL, Config  # noqa: E402
from earnings_classifier.data_collection import EarningsEvent  # noqa: E402
from earnings_classifier.providers import DocumentRef, FMPTranscripts, Headline  # noqa: E402
from earnings_classifier.timing import ET  # noqa: E402


def make_event(i: int = 0, *, day: date = date(2024, 1, 2), data_mode: str = REAL,
               label: int = 1, **overrides) -> EarningsEvent:
    """A valid event announced AMC on `day`, entering the next weekday."""
    a = datetime.combine(day, datetime.min.time().replace(hour=16, minute=5), tzinfo=ET)
    cutoff = datetime.combine(day, datetime.min.time(), tzinfo=ET)
    entry = day + timedelta(days=1 if day.weekday() < 4 else 7 - day.weekday())
    ret = 0.02 if label == 1 else -0.02
    fields = dict(
        event_id=f"ACME{i}:{day.isoformat()}", ticker=f"ACME{i}", data_mode=data_mode,
        announcement_time=a.isoformat(), session="AMC",
        document=f"Prepared remarks for event {i} on {day}.",
        document_type="earnings_call_transcript", document_source="fmp", document_ref="FY2024Q1",
        document_date=day.isoformat(), document_time="", fiscal_year=2024, fiscal_quarter=1,
        headlines=[f"ACME{i} to report"],
        headline_times=[(cutoff - timedelta(hours=2)).isoformat()],
        news_cutoff=cutoff.isoformat(), entry_date=entry.isoformat(),
        exit_date=(entry + timedelta(days=2)).isoformat(),
        prediction_time=datetime.combine(entry, datetime.min.time().replace(hour=9, minute=30),
                                         tzinfo=ET).isoformat(),
        entry_price=100.0, exit_price=100.0 * (1 + ret), forward_return=ret, label=label,
    )
    fields.update(overrides)
    return EarningsEvent(**fields)


def make_bars(start: date, end: date, holidays=(), drift: float = 0.0) -> pd.DataFrame:
    days = [d for d in pd.bdate_range(start, end) if d.date() not in set(holidays)]
    idx = pd.DatetimeIndex(days).tz_localize(ET)
    opens = [100.0 + i for i in range(len(idx))]
    closes = [o * (1 + drift) + 0.5 for o in opens]
    return pd.DataFrame({"Open": opens, "Close": closes}, index=idx)


class FakeMarket:
    source = "yahoo"

    def __init__(self, times: dict, bars: dict, fail: dict | None = None):
        self.times, self.bars, self.fail = times, bars, fail or {}

    def earnings_times(self, ticker, limit):
        if ticker in self.fail:
            raise self.fail[ticker]
        return sorted(self.times.get(ticker, []))[-limit:]

    def daily_bars(self, ticker, start, end):
        return self.bars.get(ticker)


def fmp_ref(fiscal_year: int, quarter: int, call_date: date) -> DocumentRef:
    """An FMP-style transcript reference (provider fiscal period, no exact time)."""
    return FMPTranscripts.ref(fiscal_year, quarter, call_date)


def sec_ref(accession: str, accepted: datetime) -> DocumentRef:
    """An EDGAR-style press-release reference (exact acceptance time)."""
    return DocumentRef(accession, accepted.astimezone(ET).date(),
                       accepted.astimezone(timezone.utc))


class FakeDocuments:
    def __init__(self, refs: dict, texts: dict | None = None, source: str = "fmp",
                 document_type: str = "earnings_call_transcript"):
        self.refs = refs              # ticker -> [DocumentRef]
        self.texts = texts or {}      # (ticker, ref.key) -> text
        self.source, self.document_type = source, document_type
        self.calls = []

    def list_documents(self, ticker, since=None):
        return [r for r in self.refs.get(ticker, []) if since is None or r.doc_date >= since]

    def fetch_text(self, ticker, ref):
        self.calls.append((ticker, ref))
        return self.texts.get((ticker, ref.key), f"{ticker} {ref.key} earnings document.")


class FakeNews:
    source = "finnhub"

    def __init__(self, items: dict):
        self.items = items            # ticker -> [Headline]
        self.calls = []

    def company_news(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        return [h for h in self.items.get(ticker, [])
                if start <= h.published.astimezone(ET).date() <= end]


class FakeProviders:
    def __init__(self, market, documents=None, news=None):
        self.market, self.documents, self.news = market, documents, news


def et(y, m, d, hh=0, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def headline(text: str, when: datetime) -> Headline:
    return Headline(text, when.astimezone(timezone.utc), "test")


def real_cfg(**kw) -> Config:
    base = dict(data_mode=REAL, tickers=["ACME"], min_real_events=1, earnings_per_ticker=12,
                min_document_chars=10)
    base.update(kw)
    return Config.tiny(**base)


__all__ = ["FakeEncoder", "FakeTokenizer", "FakeMarket", "FakeDocuments", "FakeNews",
           "FakeProviders", "fmp_ref", "sec_ref", "make_event", "make_bars", "et", "headline",
           "real_cfg"]
