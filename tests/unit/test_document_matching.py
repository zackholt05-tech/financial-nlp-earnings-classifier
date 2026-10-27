"""Earnings documents must match the right event, once, without leaking timing."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from conftest import FakeDocuments, FakeMarket, FakeNews, FakeProviders, et, fmp_ref, headline, \
    make_bars, real_cfg, sec_ref

from earnings_classifier.data_collection import collect_real_events, match_document, \
    validate_events
from earnings_classifier.providers import FMPTranscripts, ProviderError


def test_fiscal_quarter_comes_from_provider_not_calendar():
    # A June fiscal year: results released 2024-01-30 are fiscal Q2 of FY2024,
    # whereas the calendar quarter of the release month would say Q1.
    refs = [fmp_ref(2024, 1, date(2023, 10, 24)), fmp_ref(2024, 2, date(2024, 1, 30)),
            fmp_ref(2024, 3, date(2024, 4, 25))]
    ref, reason = match_document(date(2024, 1, 30), refs, set(), window_days=3)
    assert (ref.fiscal_year, ref.fiscal_quarter) == (2024, 2) and reason == ""


def test_nearest_document_date_within_window_wins():
    refs = [fmp_ref(2024, 2, date(2024, 2, 2)), fmp_ref(2024, 9, date(2024, 2, 1))]
    ref, _ = match_document(date(2024, 2, 1), refs, set(), window_days=3)
    assert ref.fiscal_quarter == 9


def test_same_day_tie_prefers_the_earliest_release():
    a, b = sec_ref("0000000001-24-000002", et(2024, 2, 1, 16, 30)), \
        sec_ref("0000000001-24-000001", et(2024, 2, 1, 7, 0))
    ref, _ = match_document(date(2024, 2, 1), [a, b], set(), window_days=3)
    assert ref.key == "0000000001-24-000001"


def test_document_one_day_before_release_is_allowed_but_far_dates_are_not():
    ref, _ = match_document(date(2024, 2, 1), [fmp_ref(2024, 1, date(2024, 1, 31))],
                            set(), window_days=3)
    assert ref is not None
    ref, reason = match_document(date(2024, 2, 1), [fmp_ref(2024, 1, date(2024, 2, 9))],
                                 set(), window_days=3)
    assert ref is None and reason == "no_document_in_window"


def test_document_already_used_is_rejected():
    refs = [fmp_ref(2024, 2, date(2024, 2, 1))]
    ref, reason = match_document(date(2024, 2, 2), refs, {"FY2024Q2"}, window_days=3)
    assert ref is None and reason == "document_already_matched"


def _providers(times, docs, news=None):
    return FakeProviders(
        FakeMarket({"ACME": times}, {"ACME": make_bars(date(2024, 6, 1), date(2024, 8, 30))}),
        docs, FakeNews({"ACME": news}) if news is not None else None)


def test_collector_never_attaches_one_document_to_two_events():
    times = [et(2024, 7, 10, 16, 0), et(2024, 7, 11, 16, 0)]   # e.g. a re-dated duplicate
    providers = _providers(times, FakeDocuments({"ACME": [fmp_ref(2024, 2, date(2024, 7, 10))]}))
    events, report = collect_real_events(real_cfg(), providers)
    assert [e.event_id for e in events] == ["ACME:2024-07-10"]
    assert report["document_missing"] == {"document_already_matched": 1}
    assert report["excluded"] == {"no_text": 1}


def test_press_release_fields_and_report():
    a = et(2024, 7, 10, 16, 0)
    ref = sec_ref("0000000001-24-000123", et(2024, 7, 10, 16, 31))
    docs = FakeDocuments({"ACME": [ref]}, source="sec-edgar",
                         document_type="earnings_press_release")
    events, report = collect_real_events(real_cfg(), _providers([a], docs))
    (e,) = events
    assert (e.document_source, e.document_type, e.document_ref) == \
        ("sec-edgar", "earnings_press_release", "0000000001-24-000123")
    assert e.document_date == "2024-07-10" and e.document_time.startswith("2024-07-10T20:31")
    assert e.fiscal_year is None and e.entry_date == "2024-07-11"
    assert report["providers"] == {"documents": "sec-edgar",
                                   "document_type": "earnings_press_release",
                                   "headlines": None}
    validate_events(events, real_cfg())


def test_release_before_announcement_day_moves_the_news_cutoff_earlier():
    # Yahoo stamps the 11th, but the 8-K was accepted the previous evening:
    # headlines after the release must not enter the features.
    a = et(2024, 7, 11, 16, 0)
    released = et(2024, 7, 10, 20, 30)
    news = [headline("before release", released - timedelta(hours=1)),
            headline("after release", released + timedelta(minutes=30))]
    docs = FakeDocuments({"ACME": [sec_ref("0000000001-24-000200", released)]},
                         source="sec-edgar", document_type="earnings_press_release")
    (e,), _ = collect_real_events(real_cfg(), _providers([a], docs, news))
    assert e.headlines == ["before release"]
    assert datetime.fromisoformat(e.news_cutoff) == released
    assert e.document_date == "2024-07-10" and e.entry_date == "2024-07-12"


def test_short_and_empty_documents_count_as_missing():
    times = [et(2024, 7, 10, 16, 0), et(2024, 8, 7, 16, 0)]
    refs = [fmp_ref(2024, 2, date(2024, 7, 10)), fmp_ref(2024, 3, date(2024, 8, 7))]
    docs = FakeDocuments({"ACME": refs}, texts={("ACME", "FY2024Q2"): "see exhibit",
                                                ("ACME", "FY2024Q3"): "   "})
    news = [headline("preview", t - timedelta(days=1)) for t in times]
    events, report = collect_real_events(real_cfg(min_document_chars=50),
                                         _providers(times, docs, news))
    assert [e.document for e in events] == ["", ""]
    assert [e.document_ref for e in events] == ["", ""]
    assert report["document_missing"] == {"document_too_short": 1, "empty_document": 1}


def test_validation_rejects_document_time_off_its_date():
    from conftest import make_event

    from earnings_classifier.data_collection import DataValidationError

    e = make_event(document_time="2024-01-03T20:00:00+00:00")   # document_date is Jan 2
    with pytest.raises(DataValidationError, match="document_time"):
        validate_events([e], real_cfg())


def test_validation_rejects_one_ref_on_two_events():
    from conftest import make_event

    from earnings_classifier.data_collection import DataValidationError

    a, b = make_event(0), make_event(1, day=date(2024, 4, 2))
    b.ticker = a.ticker
    with pytest.raises(DataValidationError, match="document ref"):
        validate_events([a, b], real_cfg())


class _Http:
    def __init__(self, body):
        self.body = body

    def get(self, url, params):
        return self.body


def test_fmp_payload_with_mismatched_date_is_an_error():
    fmp = FMPTranscripts(_Http([{"symbol": "ACME", "period": "Q2", "year": 2024,
                                 "date": "2024-04-25 17:00:00", "content": "text"}]))
    with pytest.raises(ProviderError, match="does not match"):
        fmp.fetch_text("ACME", fmp_ref(2024, 2, date(2024, 1, 30)))


def test_fmp_dates_schema_mismatch_is_an_error():
    with pytest.raises(ProviderError, match="unexpected transcript-dates row"):
        FMPTranscripts(_Http([{"year": 2024, "date": "2024-01-30"}])).list_documents("ACME")


def test_fmp_parses_documented_fields():
    fmp = FMPTranscripts(_Http([{"quarter": 2, "fiscalYear": 2024, "date": "2024-01-30"}]))
    assert fmp.list_documents("ACME") == [fmp_ref(2024, 2, date(2024, 1, 30))]
    fmp.http = _Http([{"symbol": "ACME", "period": "Q2", "year": 2024, "date": "2024-01-30",
                       "content": " Hello. "}])
    assert fmp.fetch_text("ACME", fmp_ref(2024, 2, date(2024, 1, 30))) == "Hello."
    fmp.http = _Http([])
    assert fmp.fetch_text("ACME", fmp_ref(2024, 2, date(2024, 1, 30))) == ""
