"""Chronological, date-grouped, purged train/val/test protocol."""

from __future__ import annotations

import random
from collections import defaultdict
from datetime import date, timedelta

import pytest
from conftest import make_event

from earnings_classifier.config import NEGATIVE_CONTROL, Config
from earnings_classifier.data_collection import generate_synthetic_events, temporal_split


def _events(n_days=30, per_day=3, horizon_gap=2):
    out, d, i = [], date(2024, 1, 1), 0
    for _ in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        for _ in range(per_day):
            e = make_event(i, day=d, label=i % 2)
            e.exit_date = (e.entry_day + timedelta(days=horizon_gap)).isoformat()
            out.append(e)
            i += 1
        d += timedelta(days=1)
    return out


def test_same_date_groups_stay_in_one_split():
    events, rep = temporal_split(_events(), Config.tiny())
    val_start = date.fromisoformat(rep["val_start"])
    test_start = date.fromisoformat(rep["test_start"])

    def side(d):  # which split a prediction date belongs to, before purging
        return "train" if d < val_start else "val" if d < test_start else "test"

    by_date = defaultdict(set)
    for e in events:
        if e.split != "purged":
            assert e.split == side(e.entry_day)
            by_date[e.entry_date].add(e.split)
    assert all(len(s) == 1 for s in by_date.values())
    assert max(len([e for e in events if e.entry_date == d]) for d in by_date) == 3


def test_label_windows_never_cross_into_later_splits():
    events, rep = temporal_split(_events(horizon_gap=4), Config.tiny())
    val_start, test_start = date.fromisoformat(rep["val_start"]), date.fromisoformat(rep["test_start"])
    for e in events:
        if e.split == "train":
            assert e.exit_day < val_start
        if e.split == "val":
            assert val_start <= e.entry_day and e.exit_day < test_start
        if e.split == "test":
            assert e.entry_day >= test_start
    assert rep["purged"]["train"] > 0 and rep["purged"]["val"] > 0
    assert sum(r["n"] for r in rep["splits"].values()) + sum(rep["purged"].values()) == len(events)


def test_explicit_boundaries():
    cfg = Config.tiny(val_start_date="2024-01-22", test_start_date="2024-02-01")
    events, rep = temporal_split(_events(), cfg)
    assert rep["val_start"] == "2024-01-22" and rep["test_start"] == "2024-02-01"
    assert all(e.entry_day >= date(2024, 2, 1) for e in events if e.split == "test")


def test_boundaries_must_be_ordered_and_paired():
    with pytest.raises(ValueError, match="before"):
        temporal_split(_events(), Config.tiny(val_start_date="2024-02-01",
                                              test_start_date="2024-01-22"))
    with pytest.raises(ValueError, match="both"):
        temporal_split(_events(), Config.tiny(val_start_date="2024-02-01"))


def test_too_few_dates_raises():
    with pytest.raises(ValueError):
        temporal_split(_events(n_days=1, per_day=5), Config.tiny())


def test_split_is_deterministic_under_input_order():
    events = _events()
    a, _ = temporal_split(list(events), Config.tiny())
    snapshot = {e.event_id: e.split for e in a}
    shuffled = list(events)
    random.Random(0).shuffle(shuffled)
    b, _ = temporal_split(shuffled, Config.tiny())
    assert {e.event_id: e.split for e in b} == snapshot


def test_synthetic_controls_exercise_grouping_and_purging():
    cfg = Config.tiny(n_synthetic_events=200, data_mode=NEGATIVE_CONTROL)
    events = generate_synthetic_events(cfg)
    assert len({e.entry_date for e in events}) < len(events)   # same-date clusters exist
    _, rep = temporal_split(events, cfg)
    assert sum(rep["purged"].values()) > 0
