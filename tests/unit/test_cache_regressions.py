"""Regression tests for the three reproduced cache bugs.

The pre-fix versions of these tests (commit ad67ae9) failed on the original
code; this file states the same scenarios against the fixed API.
  1. collect_events served a cached dataset regardless of the requested mode.
  2. compute_embeddings accepted any cache with the same model name + count.
  3. --no-cache bypassed the event cache but not the embedding cache.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from conftest import FakeMarket, FakeProviders, real_cfg

import earnings_classifier.data_collection as dc
from earnings_classifier.config import NEGATIVE_CONTROL, POSITIVE_CONTROL, Config
from earnings_classifier.embeddings import compute_embeddings

CPU = torch.device("cpu")
SIGNAL = set(dc._BULLISH_SENTENCES) | set(dc._BEARISH_SENTENCES)


def _cfg(mode: str, **kw) -> Config:
    return Config.tiny(**{"n_synthetic_events": 40, "data_mode": mode, **kw})


def _split(events, cfg):
    events, _ = dc.temporal_split(events, cfg)
    return [e for e in events if e.split != "purged"]


# ---- Bug 1: events cache is mode-blind -------------------------------------
def test_real_request_never_served_cached_synthetic(tmp_path):
    cache = str(tmp_path)
    dc.collect_events(_cfg(NEGATIVE_CONTROL), cache_dir=cache)
    no_text = FakeProviders(FakeMarket({}, {}))  # no document/news provider configured
    with pytest.raises(dc.RealDataError, match="text provider"):
        dc.collect_events(real_cfg(), cache_dir=cache, providers=no_text)


def test_legacy_events_json_is_never_read(tmp_path):
    legacy = [{"ticker": "AAPL", "earnings_date": "2021-01-05", "label": 1, "is_synthetic": True}]
    (tmp_path / "events.json").write_text(json.dumps(legacy))
    events, info = dc.collect_events(_cfg(NEGATIVE_CONTROL), cache_dir=str(tmp_path))
    assert not info["cache_hit"]
    assert all(e.data_mode == NEGATIVE_CONTROL for e in events)


def test_control_modes_are_isolated(tmp_path):
    cache = str(tmp_path)
    neg, _ = dc.collect_events(_cfg(NEGATIVE_CONTROL), cache_dir=cache)
    pos, info = dc.collect_events(_cfg(POSITIVE_CONTROL), cache_dir=cache)
    assert not info["cache_hit"]
    assert all(any(s in e.document for s in SIGNAL) for e in pos)
    assert not any(any(s in e.document for s in SIGNAL) for e in neg)
    assert len(list(tmp_path.glob("events-*.json"))) == 2


def test_same_request_hits_and_changed_settings_miss(tmp_path):
    cache = str(tmp_path)
    first, _ = dc.collect_events(_cfg(NEGATIVE_CONTROL), cache_dir=cache)
    again, info = dc.collect_events(_cfg(NEGATIVE_CONTROL), cache_dir=cache)
    assert info["cache_hit"]
    assert [e.content_dict() for e in again] == [e.content_dict() for e in first]
    for change in ({"seed": 7}, {"n_synthetic_events": 41}, {"horizon_days": 2},
                   {"tickers": ["ACME"]}, {"max_headlines": 3}):
        _, info = dc.collect_events(_cfg(NEGATIVE_CONTROL, **change), cache_dir=cache)
        assert not info["cache_hit"], change


def test_tampered_events_cache_is_an_error(tmp_path):
    cfg = _cfg(NEGATIVE_CONTROL)
    _, info = dc.collect_events(cfg, cache_dir=str(tmp_path))
    payload = json.loads(open(info["cache_path"]).read())
    payload["events"][0]["label"] = 1 - payload["events"][0]["label"]
    open(info["cache_path"], "w").write(json.dumps(payload))
    from earnings_classifier.cache import CacheError
    with pytest.raises(CacheError, match="fingerprint"):
        dc.collect_events(cfg, cache_dir=str(tmp_path))


def test_splits_are_not_cached(tmp_path):
    events, _ = dc.collect_events(_cfg(NEGATIVE_CONTROL), cache_dir=str(tmp_path))
    assert {e.split for e in events} == {""}


# ---- Bug 2: embedding cache keyed only on model name + count ----------------
def test_embedding_cache_rejects_changed_labels(tmp_path, fake_models):
    cfg = _cfg(NEGATIVE_CONTROL)
    events = _split(dc.collect_events(cfg)[0], cfg)
    compute_embeddings(events, cfg, CPU, cache_dir=str(tmp_path))
    for e in events:
        e.label = 1 - e.label
    out = compute_embeddings(events, cfg, CPU, cache_dir=str(tmp_path))
    assert not out["cache_hit"]
    assert (out["labels"] == np.array([e.label for e in events])).all()


def test_embedding_cache_rejects_changed_text(tmp_path, fake_models):
    cfg = _cfg(NEGATIVE_CONTROL)
    events = _split(dc.collect_events(cfg)[0], cfg)
    first = compute_embeddings(events, cfg, CPU, cache_dir=str(tmp_path))
    for e in events:
        e.document = "completely different text " + e.ticker
    second = compute_embeddings(events, cfg, CPU, cache_dir=str(tmp_path))
    assert not second["cache_hit"]
    assert not np.allclose(first["t_emb"], second["t_emb"])


def test_embedding_cache_hits_only_for_identical_request(tmp_path, fake_models):
    cfg = _cfg(NEGATIVE_CONTROL)
    events = _split(dc.collect_events(cfg)[0], cfg)
    first = compute_embeddings(events, cfg, CPU, cache_dir=str(tmp_path))
    n_inst = fake_models.instances
    second = compute_embeddings(events, cfg, CPU, cache_dir=str(tmp_path))
    assert second["cache_hit"] and fake_models.instances == n_inst  # encoder not rebuilt
    np.testing.assert_array_equal(first["t_emb"], second["t_emb"])


# ---- Bug 3: --no-cache only bypassed the event cache -------------------------
def _run_main(tmp_path, monkeypatch, *extra):
    import main as main_mod

    monkeypatch.chdir(tmp_path)
    return main_mod.main(["--data-mode", NEGATIVE_CONTROL, "--preset", "tiny",
                          "--n-synthetic", "60", "--epochs", "1", "--device", "cpu",
                          "--runs-dir", str(tmp_path / "runs"), *extra])


def test_no_cache_reads_and_writes_nothing(tmp_path, monkeypatch, fake_models):
    _run_main(tmp_path, monkeypatch)                  # populate both caches
    cache_files = sorted(os.listdir(tmp_path / "data" / "cache"))
    assert any(f.startswith("events-") for f in cache_files)
    assert any(f.startswith("embeddings-") for f in cache_files)

    fake_models.instances = 0
    run_dir = _run_main(tmp_path, monkeypatch, "--no-cache")
    assert fake_models.instances > 0, "--no-cache loaded cached embeddings"
    assert sorted(os.listdir(tmp_path / "data" / "cache")) == cache_files  # nothing written
    manifest = json.loads(open(os.path.join(run_dir, "manifest.json")).read())
    assert manifest["events_cache"]["cache_hit"] is False
    assert manifest["embeddings_cache"]["hit"] is False


def test_second_cached_run_hits_both_caches(tmp_path, monkeypatch, fake_models):
    _run_main(tmp_path, monkeypatch)
    fake_models.instances = 0
    run_dir = _run_main(tmp_path, monkeypatch)
    manifest = json.loads(open(os.path.join(run_dir, "manifest.json")).read())
    assert manifest["events_cache"]["cache_hit"] and manifest["embeddings_cache"]["hit"]
    assert fake_models.instances == 0


def test_embedding_cache_is_device_specific():
    from earnings_classifier.cache import embeddings_cache_key

    cfg = _cfg(NEGATIVE_CONTROL)
    events = _split(dc.collect_events(cfg)[0], cfg)
    assert embeddings_cache_key(events, cfg, "cpu")[0] != embeddings_cache_key(events, cfg, "mps")[0]


@pytest.mark.parametrize("use_cache,expect_raw", [(True, True), (False, False)])
def test_real_mode_raw_provider_cache_follows_read_policy(tmp_path, monkeypatch, use_cache,
                                                          expect_raw):
    """--no-cache / --refresh-cache (use_cache=False) must not touch data/raw."""
    from earnings_classifier import providers as prov

    seen = {}

    def fake_from_env(raw_cache_dir, documents, headlines):
        seen.update(raw=raw_cache_dir, documents=documents, headlines=headlines)
        return FakeProviders(FakeMarket({}, {}))   # no text providers -> RealDataError

    monkeypatch.setattr(prov.Providers, "from_env", staticmethod(fake_from_env))
    with pytest.raises(dc.RealDataError):
        dc.collect_events(real_cfg(data_dir=str(tmp_path), document_provider="fmp"),
                          cache_dir=None, use_cache=use_cache)
    assert (seen["raw"] is not None) == expect_raw
    assert (seen["documents"], seen["headlines"]) == ("fmp", "finnhub")   # from the config
