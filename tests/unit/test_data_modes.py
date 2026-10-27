"""Explicit data modes; a real-data request never silently becomes synthetic."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest
from conftest import FakeMarket, FakeNews, FakeProviders, et, headline, make_bars, real_cfg

import earnings_classifier.data_collection as dc
from earnings_classifier.config import (
    DATA_MODES,
    NEGATIVE_CONTROL,
    POSITIVE_CONTROL,
    REAL,
    Config,
)
from earnings_classifier.providers import ProviderAuthError, ProviderError

SIGNAL = set(dc._BULLISH_SENTENCES) | set(dc._BEARISH_SENTENCES)
BARS = {"ACME": make_bars(date(2024, 1, 1), date(2024, 12, 31))}


def _news_for(times):
    return FakeNews({"ACME": [headline(f"preview {i}", t - timedelta(days=1))
                              for i, t in enumerate(times)]})


def test_real_mode_without_text_providers_fails_loudly():
    with pytest.raises(dc.RealDataError, match="FMP_API_KEY"):
        dc.collect_events(real_cfg(), providers=FakeProviders(FakeMarket({}, {})))


def test_real_mode_with_too_few_events_fails_with_coverage_report():
    times = [et(2024, 2, 1, 16, 0), et(2024, 5, 2, 16, 0)]
    providers = FakeProviders(FakeMarket({"ACME": times}, BARS), news=_news_for(times))
    with pytest.raises(dc.RealDataError) as err:
        dc.collect_events(real_cfg(min_real_events=60), providers=providers)
    assert err.value.report["usable_events"] == 2
    assert err.value.report["modality"]["headlines_only"] == 2
    assert "No synthetic substitute" in str(err.value)


def test_real_mode_success_returns_only_real_events():
    times = [et(2024, 2, 1, 16, 0), et(2024, 5, 2, 16, 0), et(2024, 8, 1, 16, 0)]
    providers = FakeProviders(FakeMarket({"ACME": times}, BARS), news=_news_for(times))
    events, info = dc.collect_events(real_cfg(min_real_events=3), providers=providers)
    assert len(events) == 3 and all(e.data_mode == REAL and not e.is_synthetic for e in events)
    assert info["report"]["modality"] == {"both": 0, "document_only": 0,
                                          "headlines_only": 3, "neither": 0}


def test_cache_hit_still_enforces_min_real_events(tmp_path):
    times = [et(2024, 2, 1, 16, 0), et(2024, 5, 2, 16, 0), et(2024, 8, 1, 16, 0)]
    providers = FakeProviders(FakeMarket({"ACME": times}, BARS), news=_news_for(times))
    dc.collect_events(real_cfg(min_real_events=3), cache_dir=str(tmp_path), providers=providers)
    with pytest.raises(dc.RealDataError, match="Cached real dataset"):
        dc.collect_events(real_cfg(min_real_events=60), cache_dir=str(tmp_path),
                          providers=providers)


def test_events_without_any_text_are_excluded_and_counted():
    times = [et(2024, 2, 1, 16, 0), et(2024, 5, 2, 16, 0)]
    news = FakeNews({"ACME": [headline("only one", times[0] - timedelta(days=1))]})
    events, report = dc.collect_real_events(real_cfg(),
                                            FakeProviders(FakeMarket({"ACME": times}, BARS),
                                                          news=news))
    assert len(events) == 1 and report["excluded"] == {"no_text": 1}


def test_auth_errors_stop_the_run():
    market = FakeMarket({}, {}, fail={"ACME": ProviderAuthError("fmp refused: plan")})
    with pytest.raises(ProviderAuthError):
        dc.collect_real_events(real_cfg(), FakeProviders(market, news=FakeNews({})))


def test_other_provider_errors_are_recorded_not_swallowed():
    times = [et(2024, 2, 1, 16, 0)]
    market = FakeMarket({"ACME": times}, BARS, fail={"BAD": ProviderError("yahoo: boom")})
    events, report = dc.collect_real_events(real_cfg(tickers=["BAD", "ACME"]),
                                            FakeProviders(market, news=_news_for(times)))
    assert len(events) == 1
    assert report["errors"] == ["BAD: yahoo: boom"]


@pytest.mark.parametrize("mode", [NEGATIVE_CONTROL, POSITIVE_CONTROL])
def test_synthetic_modes_are_labelled_and_distinct(mode):
    events, _ = dc.collect_events(Config.tiny(data_mode=mode, n_synthetic_events=30))
    assert all(e.data_mode == mode and e.is_synthetic for e in events)
    has_signal = [any(s in e.document for s in SIGNAL) for e in events]
    assert all(has_signal) if mode == POSITIVE_CONTROL else not any(has_signal)


def test_controls_share_the_same_timeline():
    # Timing uses its own RNG stream, so both controls get identical dates/sessions.
    neg = dc.generate_synthetic_events(Config.tiny(data_mode=NEGATIVE_CONTROL,
                                                   n_synthetic_events=50))
    pos = dc.generate_synthetic_events(Config.tiny(data_mode=POSITIVE_CONTROL,
                                                   n_synthetic_events=50))
    assert [(e.entry_date, e.session) for e in neg] == [(e.entry_date, e.session) for e in pos]


def test_generator_refuses_real_mode():
    with pytest.raises(ValueError):
        dc.generate_synthetic_events(Config.tiny(data_mode=REAL))


def test_invalid_mode_and_revision_rejected():
    with pytest.raises(ValueError, match="data_mode"):
        Config(data_mode="synthetic")
    with pytest.raises(ValueError, match="commit hash"):
        Config(model_revision="main")


def test_legacy_config_maps_to_explicit_mode():
    legacy = json.load(open("runs/legacy-2026-06-18-positive-control/artifacts/config.json"))
    cfg = Config.from_dict(legacy)
    assert cfg.data_mode == POSITIVE_CONTROL and cfg.model_revision
    assert "transcript_chunk_len" in legacy                  # pre-rename field names
    assert (cfg.document_chunk_len, cfg.max_document_chunks) == (384, 8)
    assert Config.from_dict({"synthetic": True}).data_mode == NEGATIVE_CONTROL
    assert Config.from_dict({}).data_mode == REAL
    with pytest.raises(ValueError, match="unknown"):
        Config.from_dict({"not_a_field": 1})


@pytest.mark.parametrize("argv,mode", [
    ([], REAL),
    (["--synthetic"], NEGATIVE_CONTROL),
    (["--synthetic-signal"], POSITIVE_CONTROL),
    (["--synthetic", "--synthetic-signal"], POSITIVE_CONTROL),
    (["--data-mode", NEGATIVE_CONTROL], NEGATIVE_CONTROL),
])
def test_cli_modes_and_deprecated_aliases(argv, mode):
    import main as main_mod

    assert main_mod.parse_args(argv).data_mode == mode
    assert set(DATA_MODES) >= {mode}


def test_cli_rejects_conflicting_mode_flags():
    import main as main_mod

    with pytest.raises(SystemExit):
        main_mod.parse_args(["--synthetic", "--data-mode", POSITIVE_CONTROL])
    with pytest.raises(SystemExit):
        main_mod.parse_args(["--no-cache", "--refresh-cache"])
    with pytest.raises(SystemExit):
        main_mod.parse_args(["--model", "some/model"])
