"""Information timeline: sessions, trading days, and label windows.

All market logic runs in exchange time (America/New_York). The implemented
timeline for one earnings event (see docs/PLAN.md):

  A  announcement time (provider timestamp, ET) -> session BMO / DMH / AMC / UNKNOWN
  news cutoff = 00:00 ET on date(A): only headlines published on earlier days are
               features. Provider timestamps for A are approximate (Yahoo stamps
               16:00 / 06:00), so same-day news is excluded entirely rather than
               trusting the stamped minute.
  D  information date = max(date(A), earnings-call date)
  entry day = first trading day strictly after D; prediction time = 09:30 ET that day
  label = sign(Close[exit] / Open[entry] - 1), exit = horizon-th session (entry = 1)

Trading days come from the price series itself, so exchange holidays and early
closes need no separate calendar.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

BMO, DMH, AMC, UNKNOWN = "BMO", "DMH", "AMC", "UNKNOWN"


def to_et(ts) -> datetime:
    """Timezone-aware pandas/py datetime -> aware datetime in ET. Naive input is
    rejected: guessing its zone is exactly how dates silently shift."""
    dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    if not isinstance(dt, datetime):
        raise TypeError(f"expected a datetime, got {type(dt).__name__}")
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime {dt.isoformat()} has no timezone")
    return dt.astimezone(ET)


def classify_session(ts) -> str:
    """BMO before the 09:30 open, AMC at/after the 16:00 close, DMH in between.
    An exact-midnight stamp means the provider gave a date without a time."""
    t = to_et(ts).timetz().replace(tzinfo=None)
    if t == time(0, 0):
        return UNKNOWN
    if t < MARKET_OPEN:
        return BMO
    if t >= MARKET_CLOSE:
        return AMC
    return DMH


def utc_from_unix(seconds: float) -> datetime:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc)


def news_cutoff(announcement) -> datetime:
    """Start of the announcement's ET calendar day (conservative news cutoff)."""
    return datetime.combine(to_et(announcement).date(), time(0, 0), tzinfo=ET)


def news_window(cutoff: datetime, days: int) -> tuple[date, date]:
    """Calendar-date request range for a news API covering [cutoff - days, cutoff]."""
    c = to_et(cutoff)
    return (c - timedelta(days=days)).date(), c.date()


def prediction_time(entry_day: date) -> datetime:
    return datetime.combine(entry_day, MARKET_OPEN, tzinfo=ET)


@dataclass(frozen=True)
class LabelWindow:
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float

    @property
    def ret(self) -> float:
        return self.exit_price / self.entry_price - 1.0


def label_window(
    trading_days: Sequence[date],
    opens: Sequence[float],
    closes: Sequence[float],
    info_date: date,
    horizon: int,
) -> Optional[LabelWindow]:
    """Entry at the open of the first trading day strictly after `info_date`,
    exit at the close of the `horizon`-th session. None if the window is not
    fully inside the price history or a price is missing/non-positive."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if any(b <= a for a, b in zip(trading_days, trading_days[1:])):
        raise ValueError("trading_days must be strictly increasing")
    i = bisect.bisect_right(list(trading_days), info_date)
    j = i + horizon - 1
    if j >= len(trading_days):
        return None
    p0, p1 = float(opens[i]), float(closes[j])
    if not (math.isfinite(p0) and math.isfinite(p1)) or p0 <= 0 or p1 <= 0:
        return None
    return LabelWindow(trading_days[i], trading_days[j], p0, p1)


def daily_bars_et(df) -> tuple[list, list, list]:
    """yfinance daily history -> (ET trading dates, opens, closes)."""
    idx = df.index
    if getattr(idx, "tz", None) is None:
        raise ValueError("price index must be timezone-aware")
    days = [d for d in idx.tz_convert(ET).date]
    return days, df["Open"].astype(float).tolist(), df["Close"].astype(float).tolist()


def add_business_days(d: date, n: int) -> date:
    """Weekday arithmetic (no holiday calendar); used only for synthetic events."""
    return np.busday_offset(np.datetime64(d), n, roll="forward").astype(date)
