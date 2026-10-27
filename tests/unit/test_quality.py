"""Dataset invariants: duplicates, leakage, timing order, empty text, coverage."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from conftest import make_event, real_cfg

from earnings_classifier.config import NEGATIVE_CONTROL, Config
from earnings_classifier.data_collection import (
    DataValidationError,
    assert_no_leakage,
    dataset_fingerprint,
    modality_counts,
    validate_events,
)


def _ok():
    return [make_event(i, day=date(2024, 1, 2) + timedelta(days=i)) for i in range(3)]


def test_valid_dataset_passes():
    validate_events(_ok(), real_cfg())


def test_empty_dataset_rejected():
    with pytest.raises(DataValidationError, match="empty"):
        validate_events([], real_cfg())


def test_duplicate_event_ids_rejected():
    evs = _ok()
    evs[1].event_id = evs[0].event_id
    with pytest.raises(DataValidationError, match="duplicate event ids"):
        validate_events(evs, real_cfg())


def test_post_cutoff_headline_rejected():
    evs = _ok()
    cutoff = datetime.fromisoformat(evs[0].news_cutoff)
    evs[0].headline_times = [(cutoff + timedelta(minutes=1)).isoformat()]
    with pytest.raises(DataValidationError, match="Leakage"):
        validate_events(evs, real_cfg())


def test_assert_no_leakage_direct():
    t0 = datetime.fromisoformat(make_event().news_cutoff)
    assert_no_leakage([t0 - timedelta(hours=1), t0], t0)
    with pytest.raises(DataValidationError):
        assert_no_leakage([t0 + timedelta(seconds=1)], t0)


def test_entry_must_follow_information_date():
    evs = _ok()
    evs[0].entry_date = evs[0].document_date            # same day as the call
    with pytest.raises(DataValidationError, match="not after"):
        validate_events(evs, real_cfg())
    evs = _ok()
    evs[0].document_date = evs[0].entry_date            # call after the release
    with pytest.raises(DataValidationError, match="not after"):
        validate_events(evs, real_cfg())


def test_label_must_match_return():
    evs = _ok()
    evs[0].label = 1 - evs[0].label
    with pytest.raises(DataValidationError, match="label"):
        validate_events(evs, real_cfg())


def test_real_event_without_text_rejected():
    evs = _ok()
    evs[0].document, evs[0].headlines, evs[0].headline_times = "  ", [], []
    with pytest.raises(DataValidationError, match="without any text"):
        validate_events(evs, real_cfg())


def test_duplicate_document_content_rejected_for_real_data():
    evs = _ok()
    evs[1].document = evs[0].document
    with pytest.raises(DataValidationError, match="more than one event"):
        validate_events(evs, real_cfg())


def test_mixed_data_modes_rejected():
    evs = _ok()
    evs[0].data_mode = NEGATIVE_CONTROL
    with pytest.raises(DataValidationError, match="data modes"):
        validate_events(evs, real_cfg())


def test_modality_counts():
    evs = _ok()
    evs[0].document = ""
    evs[1].headlines, evs[1].headline_times = [], []
    evs[2].document, evs[2].headlines = "", [" "]
    assert modality_counts(evs) == {"both": 0, "document_only": 1, "headlines_only": 1,
                                    "neither": 1}


def test_fingerprint_is_order_independent_and_content_sensitive():
    evs = _ok()
    fp = dataset_fingerprint(evs)
    assert dataset_fingerprint(list(reversed(evs))) == fp
    evs[0].split = "test"
    assert dataset_fingerprint(evs) == fp                  # split is not dataset content
    evs[0].headlines = ["changed"]
    assert dataset_fingerprint(evs) != fp


def test_synthetic_controls_pass_validation():
    from earnings_classifier.data_collection import generate_synthetic_events

    cfg = Config.tiny(data_mode=NEGATIVE_CONTROL, n_synthetic_events=120)
    validate_events(generate_synthetic_events(cfg), cfg)
