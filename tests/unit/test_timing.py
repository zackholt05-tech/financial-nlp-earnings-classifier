"""Information-timeline tests: sessions, time zones, trading days, label windows,
and the real-data collector's entry/news-cutoff rules (all with hand-made data)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from conftest import FakeDocuments, FakeMarket, FakeNews, FakeProviders, fmp_ref, \
    et, headline, make_bars, real_cfg

from earnings_classifier.data_collection import RealDataError, collect_real_events
from earnings_classifier.timing import (
    AMC,
    BMO,
    DMH,
    ET,
    UNKNOWN,
    classify_session,
    daily_bars_et,
    label_window,
    to_et,
)


# ---- sessions and time zones -------------------------------------------------
@pytest.mark.parametrize("hh,mm,expected", [
    (6, 0, BMO), (9, 29, BMO), (9, 30, DMH), (15, 59, DMH), (16, 0, AMC), (20, 30, AMC),
    (0, 0, UNKNOWN),
])
def test_classify_session(hh, mm, expected):
    assert classify_session(et(2024, 5, 2, hh, mm)) == expected


def test_late_evening_et_does_not_roll_to_next_utc_date():
    utc = datetime(2024, 5, 3, 0, 30, tzinfo=timezone.utc)   # 20:30 EDT on May 2
    assert to_et(utc).date() == date(2024, 5, 2)
    assert classify_session(utc) == AMC


@pytest.mark.parametrize("utc_hour,day", [(20, date(2024, 3, 8)),    # EST: 15:00 -> DMH
                                          (20, date(2024, 3, 11))])  # EDT: 16:00 -> AMC
def test_dst_transition_changes_session_for_same_utc_hour(utc_hour, day):
    ts = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=utc_hour)
    expected = DMH if day < date(2024, 3, 10) else AMC
    assert classify_session(ts) == expected


def test_dst_end_week_keeps_et_dates():
    # 2024-11-03: clocks fall back. 16:00 ET is 20:00 UTC on Friday, 21:00 UTC on Monday.
    assert classify_session(datetime(2024, 11, 1, 20, 0, tzinfo=timezone.utc)) == AMC
    assert classify_session(datetime(2024, 11, 4, 21, 0, tzinfo=timezone.utc)) == AMC
    assert classify_session(datetime(2024, 11, 4, 20, 30, tzinfo=timezone.utc)) == DMH


def test_naive_datetimes_are_rejected():
    with pytest.raises(ValueError, match="naive"):
        to_et(datetime(2024, 5, 2, 16, 0))


def test_price_index_must_be_tz_aware_and_keeps_et_dates():
    bars = make_bars(date(2024, 5, 1), date(2024, 5, 3))
    days, _, _ = daily_bars_et(bars.tz_convert("UTC"))   # 04:00 UTC == midnight EDT
    assert days == [date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3)]
    with pytest.raises(ValueError, match="timezone-aware"):
        daily_bars_et(bars.tz_localize(None))


# ---- label windows -------------------------------------------------------------
DAYS = [date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3),   # Jul 4 holiday missing
        date(2024, 7, 5), date(2024, 7, 8), date(2024, 7, 9), date(2024, 7, 10)]
OPENS = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
CLOSES = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5]


def test_entry_is_first_trading_day_strictly_after_information_date():
    w = label_window(DAYS, OPENS, CLOSES, date(2024, 7, 1), horizon=3)
    assert (w.entry_date, w.exit_date) == (date(2024, 7, 2), date(2024, 7, 5))
    assert w.entry_price == 11.0 and w.exit_price == 13.5   # open(entry) -> close(3rd session)


def test_holiday_is_skipped():
    w = label_window(DAYS, OPENS, CLOSES, date(2024, 7, 3), horizon=1)
    assert w.entry_date == date(2024, 7, 5)


def test_friday_announcement_enters_monday():
    w = label_window(DAYS, OPENS, CLOSES, date(2024, 7, 5), horizon=2)
    assert (w.entry_date, w.exit_date) == (date(2024, 7, 8), date(2024, 7, 9))


def test_window_past_end_of_data_is_unavailable():
    assert label_window(DAYS, OPENS, CLOSES, date(2024, 7, 8), horizon=3) is None
    assert label_window(DAYS, OPENS, CLOSES, date(2024, 7, 10), horizon=1) is None


def test_invalid_prices_make_window_unavailable():
    bad = list(OPENS)
    bad[1] = float("nan")
    assert label_window(DAYS, bad, CLOSES, date(2024, 7, 1), horizon=1) is None
    bad[1] = 0.0
    assert label_window(DAYS, bad, CLOSES, date(2024, 7, 1), horizon=1) is None


# ---- real collector: timeline end to end ------------------------------------------
def _collect(times, refs=None, news=None, holidays=(), **cfg_kw):
    bars = make_bars(date(2024, 4, 1), date(2024, 8, 30), holidays=holidays)
    providers = FakeProviders(
        FakeMarket({"ACME": times}, {"ACME": bars}),
        FakeDocuments({"ACME": refs}) if refs is not None else None,
        FakeNews({"ACME": news}) if news is not None else None,
    )
    return collect_real_events(real_cfg(**cfg_kw), providers)


def test_bmo_event_enters_next_session_not_same_day_open():
    a = et(2024, 7, 12, 6, 0)   # Friday pre-open release (call ~08:30, may run past 09:30)
    events, _ = _collect([a], refs=[fmp_ref(2024, 2, date(2024, 7, 12))])
    (e,) = events
    assert e.session == BMO
    assert e.entry_date == "2024-07-15" and e.prediction_time.startswith("2024-07-15T09:30")


def test_amc_event_enters_next_trading_day_skipping_holiday():
    a = et(2024, 7, 3, 16, 0)
    events, _ = _collect([a], refs=[fmp_ref(2024, 2, date(2024, 7, 3))],
                         holidays=[date(2024, 7, 4)])
    assert events[0].entry_date == "2024-07-05"


def test_call_after_announcement_date_delays_entry():
    a = et(2024, 7, 10, 16, 5)                 # release Wednesday after the close
    ref = fmp_ref(2024, 2, date(2024, 7, 11))   # call Thursday morning
    events, _ = _collect([a], refs=[ref])
    assert events[0].document_date == "2024-07-11"
    assert events[0].entry_date == "2024-07-12"


def test_news_cutoff_is_start_of_announcement_day():
    a = et(2024, 7, 10, 16, 0)
    day_start = et(2024, 7, 10, 0, 0)
    news = [headline("same day, before stamp", a - timedelta(hours=3)),
            headline("at cutoff", day_start), headline("previous evening", day_start - timedelta(hours=4)),
            headline("after", a + timedelta(minutes=1)),
            headline("way before", a - timedelta(days=10))]
    events, _ = _collect([a], news=news)
    (e,) = events
    assert e.headlines == ["at cutoff", "previous evening"]   # most recent first
    assert e.news_cutoff == day_start.isoformat()
    assert all(datetime.fromisoformat(t) <= day_start for t in e.headline_times)
    assert e.document == "" and e.document_date == ""


def test_same_day_news_only_event_has_no_text():
    a = et(2024, 7, 10, 7, 0)            # BMO; stamped time is approximate
    with pytest.raises(RealDataError) as err:
        _collect([a], news=[headline("results out", a - timedelta(minutes=30))])
    assert err.value.report["usable_events"] == 0
    assert err.value.report["excluded"] == {"no_text": 1}


def test_label_matches_open_to_close_return():
    a = et(2024, 7, 10, 16, 0)
    events, _ = _collect([a], news=[headline("x", a - timedelta(days=1))], horizon_days=3)
    e = events[0]
    bars = make_bars(date(2024, 4, 1), date(2024, 8, 30))
    entry = pd.Timestamp("2024-07-11", tz=ET)
    exit_ = pd.Timestamp("2024-07-15", tz=ET)
    expected = bars.loc[exit_, "Close"] / bars.loc[entry, "Open"] - 1
    assert e.exit_date == "2024-07-15"
    assert e.forward_return == pytest.approx(expected)
    assert e.label == int(expected > 0)
