"""Data collection, labeling, temporal splitting and dataset validation.

Produces `EarningsEvent`s for one explicit data mode (config.DATA_MODES):

  * real                        - Yahoo Finance prices + earnings timestamps, one
                                  earnings document per event (SEC EDGAR press
                                  release by default, or an FMP call transcript)
                                  and/or Finnhub headlines.
                                  Never falls back to synthetic data: failures
                                  raise `RealDataError` with a coverage report.
  * synthetic-negative-control  - labels drawn independently of the text, so a
                                  leak-free pipeline must score ~0.5 AUC.
  * synthetic-positive-control  - the text deliberately encodes the label, so a
                                  working pipeline must score far above 0.5.

The information timeline is defined in `timing.py`. In short: features are
headlines published before the announcement day (cutoff 00:00 ET on date(A),
or the document's own release time if that is earlier) plus the company's
earnings document; the prediction is made at the 09:30 ET open of the first
trading day after both are public; the label is the sign of the open-to-close
return over `horizon_days` sessions from that open.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import NEGATIVE_CONTROL, POSITIVE_CONTROL, REAL, Config
from .timing import (
    AMC,
    BMO,
    ET,
    add_business_days,
    classify_session,
    daily_bars_et,
    label_window,
    news_cutoff,
    news_window,
    prediction_time,
    to_et,
)
from .utils import get_logger

log = get_logger()

SCHEMA_VERSION = 3


class RealDataError(RuntimeError):
    """A requested real-data experiment cannot be built from the available data."""

    def __init__(self, message: str, report: Optional[Dict] = None):
        super().__init__(message)
        self.report = report or {}


class DataValidationError(ValueError):
    """The dataset violates a timing, leakage or quality invariant."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
@dataclass
class EarningsEvent:
    event_id: str                 # "TICKER:YYYY-MM-DD" (announcement date, ET)
    ticker: str
    data_mode: str
    announcement_time: str        # ISO-8601 with ET offset
    session: str                  # BMO | DMH | AMC | UNKNOWN
    document: str                 # earnings press release / call transcript text, "" if none
    document_type: str            # "earnings_press_release" | "earnings_call_transcript" |
                                  # "synthetic" | "" (none)
    document_source: str          # "sec-edgar" | "fmp" | "synthetic" | ""
    document_ref: str             # provider id: SEC accession number / "FY2024Q2" / ""
    document_date: str            # ISO ET date the document became public, "" if none
    document_time: str            # ISO-8601 UTC release time when the provider states it
    fiscal_year: Optional[int]    # provider's fiscal period (never derived from the calendar)
    fiscal_quarter: Optional[int]
    headlines: List[str]          # most recent first
    headline_times: List[str]     # ISO-8601 UTC, parallel to `headlines`
    news_cutoff: str              # ISO-8601: 00:00 ET on the announcement date, or the
                                  # document time if earlier; every headline must be <= this
    entry_date: str               # first trading day after the information date
    exit_date: str                # horizon-th trading session (entry day = 1)
    prediction_time: str          # 09:30 ET on entry_date
    entry_price: float            # open on entry_date
    exit_price: float             # close on exit_date
    forward_return: float
    label: int                    # 1 = up, 0 = down
    split: str = ""               # assigned per run by `temporal_split`; never cached

    @property
    def is_synthetic(self) -> bool:
        return self.data_mode != REAL

    @property
    def announcement_date(self) -> date:
        return to_et(datetime.fromisoformat(self.announcement_time)).date()

    @property
    def entry_day(self) -> date:
        return date.fromisoformat(self.entry_date)

    @property
    def exit_day(self) -> date:
        return date.fromisoformat(self.exit_date)

    @property
    def has_document(self) -> bool:
        return bool(self.document.strip())

    @property
    def has_headlines(self) -> bool:
        return any(h.strip() for h in self.headlines)

    def content_dict(self) -> Dict:
        """Everything except the per-run split assignment."""
        d = asdict(self)
        d.pop("split")
        return d


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def dataset_fingerprint(events: Sequence[EarningsEvent]) -> str:
    """Content hash of the dataset (all fields except split), order-independent."""
    rows = sorted((e.content_dict() for e in events), key=lambda d: d["event_id"])
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------
def assert_no_leakage(headline_times: List[datetime], t0: datetime) -> None:
    """Raise if any headline is timestamped after the news cutoff."""
    for ts in headline_times:
        if ts > t0:
            raise DataValidationError(
                f"Leakage: headline at {ts.isoformat()} is after cutoff {t0.isoformat()}. "
                "Post-cutoff news must never enter the features."
            )


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------
def label_from_return(ret: float, cfg: Config) -> Optional[int]:
    if abs(ret) <= cfg.deadband:
        return None  # ambiguous, drop
    return int(ret > cfg.label_threshold)


def match_document(announce: date, refs, used: set, window_days: int):
    """Pick the provider document whose public date is closest to the
    announcement, within [announce - 1 day, announce + window_days].

    Returns (ref, reason). Documents are identified by the provider's own key
    (SEC accession number, or FMP fiscal year/quarter); the calendar quarter of
    the announcement is never used (fiscal years differ from calendar years and
    results are reported a quarter late). A document already attached to an
    earlier event is never reused.
    """
    lo, hi = announce - timedelta(days=1), announce + timedelta(days=window_days)
    cands = [r for r in refs if lo <= r.doc_date <= hi]
    if not cands:
        return None, "no_document_in_window"
    best = min(cands, key=lambda r: (abs((r.doc_date - announce).days), r.doc_date < announce,
                                     r.doc_time or datetime.min.replace(tzinfo=ET)))
    if best.key in used:
        return None, "document_already_matched"
    return best, ""


def collect_real_events(cfg: Config, providers) -> Tuple[List[EarningsEvent], Dict]:
    """Collect labeled real events. Returns (events, coverage report)."""
    from .providers import ProviderAuthError, ProviderError, ProviderRateLimitError

    if providers.documents is None and providers.news is None:
        raise RealDataError(
            "Real-data mode needs at least one text provider: --documents sec-edgar "
            "(SEC_EDGAR_USER_AGENT) or fmp (FMP_API_KEY), and/or --headlines finnhub "
            "(FINNHUB_API_KEY). Refusing to build an empty-text dataset; use --data-mode "
            "synthetic-negative-control / synthetic-positive-control for pipeline checks."
        )
    docs = providers.documents

    excluded: Counter = Counter()
    document_missing: Counter = Counter()
    errors: List[str] = []
    seen = 0
    events: List[EarningsEvent] = []

    for ticker in cfg.tickers:
        try:
            times = providers.market.earnings_times(ticker, cfg.earnings_per_ticker)
            if not times:
                errors.append(f"{ticker}: no realized earnings dates")
                continue
            refs = docs.list_documents(ticker, since=min(times).date() - timedelta(days=1)) \
                if docs else []
            start = min(times).date() - timedelta(days=10)
            end = max(times).date() + timedelta(days=cfg.document_match_days
                                                + 3 * cfg.horizon_days + 15)
            bars = providers.market.daily_bars(ticker, start, end)
            if bars is None:
                errors.append(f"{ticker}: no price history")
                continue
            days, opens, closes = daily_bars_et(bars)

            used_refs: set = set()
            seen_ids: set = set()
            for a in times:
                seen += 1
                event_id = f"{ticker}:{a.date().isoformat()}"
                if event_id in seen_ids:
                    excluded["duplicate_announcement"] += 1
                    continue
                seen_ids.add(event_id)

                document, ref = "", None
                if docs:
                    ref, reason = match_document(a.date(), refs, used_refs,
                                                 cfg.document_match_days)
                    if ref is None:
                        document_missing[reason] += 1
                    else:
                        document = docs.fetch_text(ticker, ref)
                        used_refs.add(ref.key)
                        if len(document.strip()) < cfg.min_document_chars:
                            document_missing["empty_document" if not document.strip()
                                             else "document_too_short"] += 1
                            document, ref = "", None

                cutoff = news_cutoff(a)
                if ref is not None and ref.doc_time is not None and ref.doc_time < cutoff:
                    cutoff = to_et(ref.doc_time)   # document out before midnight: no later news
                heads: List[Tuple[datetime, str]] = []
                if providers.news:
                    frm, to = news_window(cutoff, cfg.news_window_days)
                    heads = sorted(
                        ((h.published, h.text) for h in providers.news.company_news(ticker, frm, to)
                         if h.published <= cutoff),       # no same-day news (see timing.py)
                        reverse=True,
                    )

                if not document and not heads:
                    excluded["no_text"] += 1
                    continue

                info_date = max(a.date(), ref.doc_date) if ref else a.date()
                win = label_window(days, opens, closes, info_date, cfg.horizon_days)
                if win is None:
                    excluded["no_label_window"] += 1
                    continue
                label = label_from_return(win.ret, cfg)
                if label is None:
                    excluded["deadband"] += 1
                    continue

                events.append(EarningsEvent(
                    event_id=event_id, ticker=ticker, data_mode=REAL,
                    announcement_time=a.isoformat(), session=classify_session(a),
                    document=document,
                    document_type=docs.document_type if ref else "",
                    document_source=docs.source if ref else "",
                    document_ref=ref.key if ref else "",
                    document_date=ref.doc_date.isoformat() if ref else "",
                    document_time=ref.doc_time.isoformat() if ref and ref.doc_time else "",
                    fiscal_year=ref.fiscal_year if ref else None,
                    fiscal_quarter=ref.fiscal_quarter if ref else None,
                    headlines=[t for _, t in heads],
                    headline_times=[ts.isoformat() for ts, _ in heads],
                    news_cutoff=cutoff.isoformat(),
                    entry_date=win.entry_date.isoformat(), exit_date=win.exit_date.isoformat(),
                    prediction_time=prediction_time(win.entry_date).isoformat(),
                    entry_price=win.entry_price, exit_price=win.exit_price,
                    forward_return=float(win.ret), label=int(label),
                ))
        except (ProviderAuthError, ProviderRateLimitError):
            raise  # systemic: wrong key / plan / quota -> stop the run
        except ProviderError as e:
            errors.append(f"{ticker}: {e}")
            log.warning("provider error for %s: %s", ticker, e)
        log.info("%s: %d usable events so far", ticker, len(events))

    report = {
        "tickers_requested": len(cfg.tickers),
        "announcements_seen": seen,
        "usable_events": len(events),
        "excluded": dict(excluded),
        "document_missing": dict(document_missing),
        "providers": {"documents": getattr(docs, "source", None),
                      "document_type": getattr(docs, "document_type", None),
                      "headlines": getattr(providers.news, "source", None)},
        "document_chars": _length_summary([len(e.document) for e in events if e.has_document]),
        "document_exhibit_types": dict(getattr(docs, "exhibit_types", {})),
        "errors": errors,
        "modality": modality_counts(events),
    }
    if len(events) < cfg.min_real_events:
        raise RealDataError(
            f"Only {len(events)} usable real events (< min_real_events={cfg.min_real_events}). "
            f"Coverage: {json.dumps(report['modality'])}; excluded: {json.dumps(report['excluded'])}; "
            f"{len(errors)} provider errors. No synthetic substitute is used.",
            report,
        )
    return events, report


def _length_summary(lengths: List[int]) -> Dict:
    if not lengths:
        return {"n": 0}
    q = np.percentile(lengths, [0, 50, 100])
    return {"n": len(lengths), "min": int(q[0]), "median": int(q[1]), "max": int(q[2])}


# ---------------------------------------------------------------------------
# Synthetic data (pipeline controls; never market evidence)
# ---------------------------------------------------------------------------
_DOCUMENT_SENTENCES = [
    "Revenue for the quarter came in line with internal expectations.",
    "Operating margin reflected ongoing investments in the platform.",
    "Management reiterated the full-year outlook provided last quarter.",
    "Free cash flow generation supported continued capital returns.",
    "The company saw mixed demand signals across its end markets.",
    "Gross margin was affected by product mix and input costs.",
    "We continue to execute against our long-term strategic priorities.",
    "Headcount remained roughly flat versus the prior period.",
    "Foreign exchange was a modest headwind to reported results.",
    "The board approved a routine quarterly dividend.",
    "Inventory levels normalized following prior-quarter actions.",
    "Bookings trends were broadly consistent with the macro backdrop.",
]
_HEADLINE_TEMPLATES = [
    "{t} to report quarterly results this week",
    "Analysts preview {t} earnings amid sector rotation",
    "{t} options imply elevated post-earnings volatility",
    "What to watch when {t} reports",
    "{t} shares little changed ahead of results",
    "Street eyes margins as {t} prepares to report",
    "{t} earnings: key metrics investors are tracking",
    "Sector peers mixed as {t} report approaches",
]


# Signal-bearing pools for the POSITIVE CONTROL: the text *encodes* the label so
# a correct pipeline must beat chance.
_BULLISH_SENTENCES = [
    "The company comfortably beat both revenue and earnings expectations.",
    "Management raised full-year guidance on accelerating demand.",
    "Margins expanded sharply and free cash flow hit a record.",
    "Bookings surged across every region and product line.",
    "We are very optimistic about strong momentum into next quarter.",
    "Customer growth accelerated and churn fell to record lows.",
    "Profitability improved well ahead of the company's own targets.",
]
_BEARISH_SENTENCES = [
    "The company missed revenue and earnings expectations badly.",
    "Management cut full-year guidance on deteriorating demand.",
    "Margins compressed sharply and cash flow turned negative.",
    "Bookings fell across every region as demand weakened.",
    "We are cautious given mounting headwinds into next quarter.",
    "Customer growth stalled and churn rose to concerning levels.",
    "Profitability deteriorated well below the company's own targets.",
]
_BULLISH_HEADLINES = [
    "{t} surges after a blowout quarter and raised guidance",
    "Analysts hike {t} targets following strong beat",
    "{t} rallies as results crush expectations",
]
_BEARISH_HEADLINES = [
    "{t} plunges after weak quarter and guidance cut",
    "Analysts slash {t} targets following big miss",
    "{t} sells off as results disappoint",
]


def generate_synthetic_events(cfg: Config) -> List[EarningsEvent]:
    """Synthetic events with the same schema and timeline as real ones.

    Labels/text use `rng` exactly as the pre-audit generator did; timing uses a
    separate stream (`rng_t`) and includes same-date clusters so the
    date-grouped split and purge logic is exercised.
    """
    if not cfg.is_synthetic:
        raise ValueError("generate_synthetic_events called for a non-synthetic data mode")
    rng = np.random.default_rng(cfg.seed)
    rng_t = np.random.default_rng([cfg.seed, 1])
    day = date(2021, 1, 5)
    n = cfg.n_synthetic_events
    events: List[EarningsEvent] = []
    used_ids: set = set()

    for i in range(n):
        ticker = cfg.tickers[i % len(cfg.tickers)]
        if i > 0 and rng_t.random() >= 0.35:   # ~35% share the previous announcement date
            day = add_business_days(day, int(rng_t.integers(1, 3)))
        session = AMC if rng_t.random() < 0.6 else BMO
        a = datetime.combine(day, datetime.strptime("16:05" if session == AMC else "07:00",
                                                    "%H:%M").time(), tzinfo=ET)

        # Label drawn INDEPENDENTLY of the text (slight up-drift, like reality).
        label = int(rng.random() < 0.53)
        mag = abs(rng.normal(0.0, 0.04)) + 0.002
        ret = mag if label == 1 else -mag

        if cfg.synthetic_signal:
            # POSITIVE CONTROL: text encodes the label. Long document (>1 chunk)
            # exercises multi-chunk pooling; sentiment drives separability.
            pool = _BULLISH_SENTENCES if label == 1 else _BEARISH_SENTENCES
            heads = _BULLISH_HEADLINES if label == 1 else _BEARISH_HEADLINES
            document = " ".join(rng.choice(pool, size=80, replace=True))
            n_head = int(rng.integers(2, cfg.max_headlines + 1))
            headlines = [rng.choice(heads).format(t=ticker) for _ in range(n_head)]
        else:
            # NEGATIVE CONTROL: text is independent of the label (-> ~0.5 AUC).
            n_sent = int(rng.integers(4, 9))
            document = " ".join(rng.choice(_DOCUMENT_SENTENCES, size=n_sent, replace=True))
            n_head = int(rng.integers(2, cfg.max_headlines + 1))
            headlines = [rng.choice(_HEADLINE_TEMPLATES).format(t=ticker) for _ in range(n_head)]

        # Fixed-size draw keeps the timing stream identical across the two controls.
        hours_pool = rng_t.integers(1, 72, size=cfg.max_headlines)
        hours_before = sorted(int(h) for h in hours_pool[: len(headlines)])
        cutoff = news_cutoff(a)
        entry = add_business_days(day, 1)
        exit_ = add_business_days(entry, cfg.horizon_days - 1)
        event_id = f"{ticker}:{day.isoformat()}"
        if event_id in used_ids:  # same ticker twice on one date: make the id unique
            event_id = f"{event_id}#{i}"
        used_ids.add(event_id)
        events.append(EarningsEvent(
            event_id=event_id, ticker=ticker, data_mode=cfg.data_mode,
            announcement_time=a.isoformat(), session=session,
            document=document, document_type="synthetic", document_source="synthetic",
            document_ref="", document_date=day.isoformat(), document_time="",
            fiscal_year=None, fiscal_quarter=None,
            headlines=headlines,
            headline_times=[(cutoff - timedelta(hours=h)).astimezone(ET).isoformat()
                            for h in hours_before],
            news_cutoff=cutoff.isoformat(),
            entry_date=entry.isoformat(), exit_date=exit_.isoformat(),
            prediction_time=prediction_time(entry).isoformat(),
            entry_price=100.0, exit_price=100.0 * (1.0 + ret),
            forward_return=float(ret), label=label,
        ))
    return events


# ---------------------------------------------------------------------------
# Validation + coverage
# ---------------------------------------------------------------------------
def modality_counts(events: Sequence[EarningsEvent]) -> Dict[str, int]:
    c = Counter()
    for e in events:
        t, h = e.has_document, e.has_headlines
        c["both" if t and h else "document_only" if t else "headlines_only" if h else "neither"] += 1
    return {k: int(c.get(k, 0)) for k in ("both", "document_only", "headlines_only", "neither")}


def validate_events(events: Sequence[EarningsEvent], cfg: Config) -> None:
    """Check the timing / leakage / quality invariants on the assembled dataset."""
    if not events:
        raise DataValidationError("dataset is empty")
    ids = Counter(e.event_id for e in events)
    dups = [k for k, v in ids.items() if v > 1]
    if dups:
        raise DataValidationError(f"duplicate event ids: {dups[:5]}")
    modes = {e.data_mode for e in events}
    if modes != {cfg.data_mode}:
        raise DataValidationError(f"events have data modes {sorted(modes)}, "
                                  f"requested {cfg.data_mode!r}")
    for e in events:
        if len(e.headlines) != len(e.headline_times):
            raise DataValidationError(f"{e.event_id}: headlines/headline_times length mismatch")
        cutoff = datetime.fromisoformat(e.news_cutoff)
        assert_no_leakage([datetime.fromisoformat(t) for t in e.headline_times], cutoff)
        info = e.announcement_date
        if e.document_date:
            info = max(info, date.fromisoformat(e.document_date))
        if e.document_time:
            t = datetime.fromisoformat(e.document_time)
            if t.tzinfo is None or to_et(t).date().isoformat() != e.document_date:
                raise DataValidationError(f"{e.event_id}: document_time {e.document_time} is "
                                          f"not on document_date {e.document_date} (ET)")
        if not e.entry_day > info:
            raise DataValidationError(f"{e.event_id}: entry {e.entry_date} is not after the "
                                      f"information date {info}")
        if e.exit_day < e.entry_day:
            raise DataValidationError(f"{e.event_id}: exit before entry")
        if e.label != label_from_return(e.forward_return, cfg):
            raise DataValidationError(f"{e.event_id}: label inconsistent with forward return")
        if cfg.data_mode == REAL and not (e.has_document or e.has_headlines):
            raise DataValidationError(f"{e.event_id}: real event without any text")
        if e.has_document != bool(e.document_date):
            raise DataValidationError(f"{e.event_id}: document text and document_date disagree")
    if cfg.data_mode == REAL:
        hashes = Counter(text_hash(e.document) for e in events if e.has_document)
        dup_t = [h for h, v in hashes.items() if v > 1]
        if dup_t:
            raise DataValidationError(f"{len(dup_t)} document(s) attached to more than one event")
        refs = Counter((e.ticker, e.document_source, e.document_ref) for e in events
                       if e.document_ref)
        dup_r = [r for r, v in refs.items() if v > 1]
        if dup_r:
            raise DataValidationError(f"document ref(s) attached to more than one event: "
                                      f"{dup_r[:3]}")


# ---------------------------------------------------------------------------
# Temporal split (date-grouped, purged)
# ---------------------------------------------------------------------------
def temporal_split(events: List[EarningsEvent], cfg: Config) -> Tuple[List[EarningsEvent], Dict]:
    """Assign train/val/test by entry (prediction) date, oldest -> train.

    * Boundaries fall between dates, so events sharing a prediction date always
      land in the same split.
    * Purge: a train event whose label window ends on/after the first
      validation date (or a val event ending on/after the first test date) is
      marked "purged" and excluded, so no training label overlaps a later split.
    """
    ordered = sorted(events, key=lambda e: (e.entry_day, e.event_id))
    n = len(ordered)
    days = sorted({e.entry_day for e in ordered})

    if cfg.val_start_date or cfg.test_start_date:
        if not (cfg.val_start_date and cfg.test_start_date):
            raise ValueError("set both val_start_date and test_start_date, or neither")
        val_start = date.fromisoformat(cfg.val_start_date)
        test_start = date.fromisoformat(cfg.test_start_date)
    else:
        def first_day_with_at_least(k: int) -> Optional[date]:
            for d in days:
                if sum(e.entry_day < d for e in ordered) >= k:
                    return d
            return None

        val_start = first_day_with_at_least(int(round(n * (1 - cfg.val_fraction - cfg.test_fraction))))
        test_start = first_day_with_at_least(int(round(n * (1 - cfg.test_fraction))))
        if val_start is None or test_start is None:
            raise ValueError(f"Not enough distinct dates ({len(days)}) / events ({n}) "
                             "for the requested val/test fractions.")
    if not val_start < test_start:
        raise ValueError(f"val_start {val_start} must be before test_start {test_start}")

    purged = Counter()
    for e in ordered:
        if e.entry_day < val_start:
            e.split = "train"
            if e.exit_day >= val_start:
                e.split, purged["train"] = "purged", purged["train"] + 1
        elif e.entry_day < test_start:
            e.split = "val"
            if e.exit_day >= test_start:
                e.split, purged["val"] = "purged", purged["val"] + 1
        else:
            e.split = "test"

    report: Dict = {"val_start": val_start.isoformat(), "test_start": test_start.isoformat(),
                    "purged": {"train": purged["train"], "val": purged["val"]}, "splits": {}}
    for s in ("train", "val", "test"):
        sub = [e for e in ordered if e.split == s]
        if not sub:
            raise ValueError(f"split {s!r} is empty after date grouping and purging")
        report["splits"][s] = {
            "n": len(sub),
            "n_pos": int(sum(e.label for e in sub)),
            "pos_rate": float(np.mean([e.label for e in sub])),
            "first_entry": sub[0].entry_date,
            "last_entry": sub[-1].entry_date,
            "last_exit": max(e.exit_date for e in sub),
            "modality": modality_counts(sub),
        }
    return ordered, report


# ---------------------------------------------------------------------------
# Frozen datasets
# ---------------------------------------------------------------------------
_FROZEN_SETTINGS = ("horizon_days", "label_threshold", "deadband", "news_window_days",
                    "document_match_days", "min_document_chars")


def load_frozen_events(path: str, expected_fingerprint: str,
                       cfg: Config) -> Tuple[List[EarningsEvent], Dict]:
    """Load a frozen events file for analysis without any provider access.

    Refuses unless the recomputed dataset fingerprint equals both the stored one
    and `expected_fingerprint`, the labeling settings match `cfg`, and the events
    pass `validate_events`. Nothing is ever re-collected."""
    from .cache import EVENTS_FORMAT

    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or payload.get("format") != EVENTS_FORMAT:
        raise DataValidationError(f"{path} is not a provenance-tagged events file")
    events = [EarningsEvent(**d) for d in payload["events"]]
    fp = dataset_fingerprint(events)
    if not (fp == payload.get("dataset_fingerprint") == expected_fingerprint):
        raise DataValidationError(
            f"{path}: fingerprint {fp} (stored {payload.get('dataset_fingerprint')}) does not "
            f"match the expected {expected_fingerprint}; refusing to use it")
    material = payload.get("key_material", {})
    if material.get("data_mode") != cfg.data_mode:
        raise DataValidationError(f"{path}: data mode {material.get('data_mode')!r} != "
                                  f"{cfg.data_mode!r}")
    diff = {k: (material.get(k), getattr(cfg, k)) for k in _FROZEN_SETTINGS
            if k in material and material[k] != getattr(cfg, k)}
    if diff:
        raise DataValidationError(f"{path}: collected with different settings {diff}")
    validate_events(events, cfg)
    return events, payload


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def collect_events(cfg: Config, cache_dir: Optional[str] = None, use_cache: bool = True,
                   write_cache: bool = True, providers=None) -> Tuple[List[EarningsEvent], Dict]:
    """Collect (or load) the events for `cfg.data_mode`. Returns (events, info).

    The cache is keyed by data mode + every collection-relevant setting (see
    cache.events_cache_key), so a request can only ever be served events that
    were produced for exactly that request. Splits are NOT part of the cache.
    """
    from .cache import events_cache_key, load_events_cache, save_events_cache
    from .providers import Providers

    if cfg.data_mode == REAL and providers is None:
        # The raw-response cache follows the read policy: --no-cache and
        # --refresh-cache both force fresh provider requests.
        providers = Providers.from_env(raw_cache_dir=f"{cfg.data_dir}/raw" if use_cache else None,
                                       documents=cfg.document_provider,
                                       headlines=cfg.headline_provider)
    key, material = events_cache_key(cfg, providers)
    path = f"{cache_dir}/events-{cfg.data_mode}-{key[:16]}.json" if cache_dir else None

    info: Dict = {"cache_key": key, "cache_path": path if (use_cache or write_cache) else None,
                  "cache_hit": False}
    if path and use_cache:
        cached = load_events_cache(path, key, material)
        if cached is not None:
            events, meta = cached
            if cfg.data_mode == REAL and len(events) < cfg.min_real_events:
                raise RealDataError(
                    f"Cached real dataset has {len(events)} events (< min_real_events="
                    f"{cfg.min_real_events}). No synthetic substitute is used.", meta["report"])
            log.info("Loaded %d cached %s events from %s", len(events), cfg.data_mode, path)
            info.update(cache_hit=True, report=meta["report"], created_at=meta["created_at"])
            return events, info

    if cfg.data_mode == REAL:
        log.info("Collecting REAL events for %d tickers...", len(cfg.tickers))
        events, report = collect_real_events(cfg, providers)
    else:
        log.info("Generating %d %s events (pipeline check, not market data).",
                 cfg.n_synthetic_events, cfg.data_mode.upper())
        events = generate_synthetic_events(cfg)
        report = {"usable_events": len(events), "modality": modality_counts(events)}
    validate_events(events, cfg)
    info["report"] = report

    if path and write_cache:
        save_events_cache(path, events, key, material, report)
        log.info("Cached %d events to %s", len(events), path)
    log.info("Collected %d %s events. Up-rate=%.3f", len(events), cfg.data_mode,
             np.mean([e.label for e in events]))
    return events, info


__all__ = [
    "SCHEMA_VERSION", "RealDataError", "DataValidationError", "EarningsEvent",
    "assert_no_leakage", "collect_events", "collect_real_events", "generate_synthetic_events",
    "match_document", "modality_counts", "temporal_split", "validate_events",
    "dataset_fingerprint", "text_hash", "label_from_return", "load_frozen_events",
    "NEGATIVE_CONTROL", "POSITIVE_CONTROL", "REAL",
]
