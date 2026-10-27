"""Week-block bootstrap, ticker prior, zero-shot scoring, selection and frozen loading."""

from __future__ import annotations

import json
import subprocess
from datetime import date, timedelta

import numpy as np
import pytest
import torch
from conftest import FakeTokenizer, make_event, real_cfg

from earnings_classifier.baselines import ticker_prior_baseline
from earnings_classifier.cache import save_events_cache
from earnings_classifier.data_collection import (
    DataValidationError,
    dataset_fingerprint,
    load_frozen_events,
)
from earnings_classifier.selection import require_committed, select_best
from earnings_classifier.uncertainty import (
    _week_index,
    draw_block_indices,
    entry_weeks,
    week_block_bootstrap,
)
from earnings_classifier.zeroshot import label_indices, sentiment_scores


# ---- week-block bootstrap ----------------------------------------------------------
def test_entry_weeks_use_iso_weeks():
    assert list(entry_weeks(["2024-12-30", "2025-01-05", "2025-01-06"])) == \
        ["2025-W01", "2025-W01", "2025-W02"]


def test_resamples_consist_of_whole_weeks():
    weeks = np.array(["a", "a", "a", "b", "c", "c"])
    blocks = _week_index(weeks)
    idx = draw_block_indices(blocks, np.random.default_rng(1))
    counts = {w: int(np.sum(weeks[idx] == w)) for w in "abc"}
    sizes = {"a": 3, "b": 1, "c": 2}
    assert all(counts[w] % sizes[w] == 0 for w in "abc")   # weeks enter whole, never split
    assert len(blocks) == 3


def _toy(n_weeks=40, per=5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n_weeks * per)
    weeks = np.repeat([f"w{i}" for i in range(n_weeks)], per)
    return y, weeks, rng


def test_perfect_and_constant_scores():
    y, weeks, rng = _toy()
    res = week_block_bootstrap(y, {"perfect": y.astype(float), "const": np.full(len(y), 0.4)},
                               weeks, [("perfect", "const")], n_boot=200)
    assert res["auc"]["perfect"]["point"] == 1.0 and res["auc"]["perfect"]["lo"] == 1.0
    assert res["auc"]["const"]["point"] == 0.5
    d = res["diff"]["perfect - const"]
    assert d["point"] == 0.5 and d["lo"] == 0.5 and d["frac_le_0"] == 0.0
    assert res["n_weeks"] == 40 and res["n_events"] == 200


def test_random_scores_interval_covers_half_and_is_deterministic():
    y, weeks, rng = _toy()
    s = rng.random(len(y))
    a = week_block_bootstrap(y, {"r": s}, weeks, n_boot=300, seed=7)
    b = week_block_bootstrap(y, {"r": s}, weeks, n_boot=300, seed=7)
    assert a == b
    assert a["auc"]["r"]["lo"] < 0.5 < a["auc"]["r"]["hi"]


def test_week_clustering_widens_intervals():
    # Labels and scores constant within a week: 40 effective observations, not 400.
    rng = np.random.default_rng(3)
    wk_y = rng.integers(0, 2, 40)
    wk_s = wk_y * 0.2 + rng.random(40)
    y, s = np.repeat(wk_y, 10), np.repeat(wk_s, 10)
    clustered = week_block_bootstrap(y, {"m": s}, np.repeat(np.arange(40), 10), n_boot=400)
    naive = week_block_bootstrap(y, {"m": s}, np.arange(400), n_boot=400)
    width = lambda r: r["auc"]["m"]["hi"] - r["auc"]["m"]["lo"]   # noqa: E731
    assert width(clustered) > 2 * width(naive)


def test_single_class_resamples_are_skipped_and_counted():
    y = np.array([0, 0, 0, 1])
    res = week_block_bootstrap(y, {"m": np.array([0.1, 0.2, 0.3, 0.9])},
                               np.array(["a", "a", "a", "b"]), n_boot=200, seed=0)
    assert res["n_skipped_single_class"] > 0


def test_bootstrap_rejects_bad_scores():
    with pytest.raises(ValueError):
        week_block_bootstrap([0, 1], {"m": [0.1, float("nan")]}, ["a", "b"])


# ---- ticker prior ------------------------------------------------------------------
def test_ticker_prior_is_shrunk_training_rate():
    train_t = ["A"] * 4 + ["B"] * 2
    y_tr = np.array([1, 1, 1, 0, 0, 0])            # global 0.5; A 3/4; B 0/2
    m, p = ticker_prior_baseline(train_t, y_tr, ["A", "B", "C"], np.array([1, 0, 1]),
                                 shrinkage=2.0)
    assert p == pytest.approx([(3 + 1.0) / 6, (0 + 1.0) / 4, 0.5])
    assert m["n"] == 3


# ---- zero-shot sentiment -----------------------------------------------------------
class _FakeSentiment(torch.nn.Module):
    """Logits favour 'positive' for documents containing the token 'beat'."""

    class config:   # noqa: N801 - mimics a transformers config attribute
        id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self):
        super().__init__()
        self.beat = FakeTokenizer()._ids("beat")[0]

    def forward(self, input_ids, attention_mask):
        pos = (input_ids == self.beat).any(dim=1).float()
        return type("Out", (), {"logits": torch.stack([pos * 4, (1 - pos) * 4,
                                                       torch.zeros_like(pos)], dim=1)})()


def test_label_indices_require_a_sentiment_head():
    assert label_indices({0: "Positive", 1: "negative", 2: "neutral"}) == \
        {"positive": 0, "negative": 1}
    with pytest.raises(ValueError, match="sentiment head"):
        label_indices({0: "LABEL_0", 1: "LABEL_1"})


def test_sentiment_scores_rank_documents_and_mark_missing_text():
    cfg = real_cfg()
    evs = [make_event(0, document="we beat estimates"), make_event(1, document="we missed"),
           make_event(2, document="", headlines=[], headline_times=[])]
    s = sentiment_scores(evs, cfg, torch.device("cpu"), "document",
                         model=_FakeSentiment(), tokenizer=FakeTokenizer())
    assert s[0] > 0.9 and s[1] < -0.9 and np.isnan(s[2])
    h = sentiment_scores(evs, cfg, torch.device("cpu"), "headlines",
                         model=_FakeSentiment(), tokenizer=FakeTokenizer())
    assert np.isfinite(h[:2]).all() and np.isnan(h[2])


# ---- selection ---------------------------------------------------------------------
def test_select_best_prefers_score_then_simplicity():
    rows = [{"chunks": 8, "layers": 1, "auc": 0.61}, {"chunks": 1, "layers": 0, "auc": 0.61},
            {"chunks": 8, "layers": 0, "auc": 0.59}]
    assert select_best(rows, "auc", ["chunks", "layers"]) == rows[1]
    rows[0]["auc"] = 0.62
    assert select_best(rows, "auc", ["chunks", "layers"]) == rows[0]
    rows[2]["auc"] = 0.6100000001                     # float noise is a tie
    rows[0]["auc"] = 0.5
    assert select_best(rows, "auc", ["chunks", "layers"]) == rows[1]
    with pytest.raises(ValueError):
        select_best([{"auc": float("nan"), "chunks": 1, "layers": 0}], "auc", ["chunks"])


def test_require_committed(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    f = tmp_path / "selection.json"
    f.write_text("{}")
    with pytest.raises(RuntimeError, match="not committed"):
        require_committed("selection.json", str(tmp_path))
    git("add", "selection.json")
    git("commit", "-q", "-m", "s")
    assert len(require_committed("selection.json", str(tmp_path))) == 40
    f.write_text('{"changed": 1}')
    with pytest.raises(RuntimeError, match="differs"):
        require_committed("selection.json", str(tmp_path))


# ---- frozen datasets ---------------------------------------------------------------
def _frozen(tmp_path, cfg):
    evs = [make_event(i, day=date(2024, 1, 2) + timedelta(days=7 * i)) for i in range(3)]
    path = tmp_path / "frozen.json"
    material = {"data_mode": cfg.data_mode, "horizon_days": cfg.horizon_days,
                "deadband": cfg.deadband}
    save_events_cache(str(path), evs, "k" * 64, material, {"usable_events": 3})
    return path, dataset_fingerprint(evs)


def test_frozen_events_load_only_with_the_expected_fingerprint(tmp_path):
    cfg = real_cfg()
    path, fp = _frozen(tmp_path, cfg)
    events, payload = load_frozen_events(str(path), fp, cfg)
    assert len(events) == 3 and payload["report"] == {"usable_events": 3}
    with pytest.raises(DataValidationError, match="expected"):
        load_frozen_events(str(path), "0" * 64, cfg)


def test_frozen_events_reject_tampering_and_other_settings(tmp_path):
    cfg = real_cfg()
    path, fp = _frozen(tmp_path, cfg)
    with pytest.raises(DataValidationError, match="different settings"):
        load_frozen_events(str(path), fp, real_cfg(horizon_days=5))
    d = json.loads(path.read_text())
    d["events"][0]["label"] = 1 - d["events"][0]["label"]
    path.write_text(json.dumps(d))
    with pytest.raises(DataValidationError, match="fingerprint"):
        load_frozen_events(str(path), fp, cfg)


# ---- runner helpers (fake embeddings, CPU) ------------------------------------------
def _fake_emb(n=60, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    t = rng.normal(size=(n, dim)).astype(np.float32)
    t[:, 0] += labels                                   # some signal
    splits = np.array(["train"] * 30 + ["val"] * 15 + ["test"] * 15)
    return {"t_emb": t, "n_emb": np.zeros_like(t), "t_present": np.ones(n, np.float32),
            "n_present": np.zeros(n, np.float32), "labels": labels, "splits": splits,
            "event_ids": np.array([f"e{i}" for i in range(n)])}


def test_runner_head_is_seed_reproducible_and_select_never_touches_test(monkeypatch):
    import importlib
    import sys

    sys.path.insert(0, ".")
    phase4 = importlib.import_module("experiments.phase4")
    import earnings_classifier.embeddings as emb_mod

    seen = []
    real = emb_mod.make_embedding_dataset
    monkeypatch.setattr(phase4, "make_embedding_dataset",
                        lambda arr, split, *a: (seen.append(split), real(arr, split, *a))[1])
    emb = _fake_emb()
    cfg = real_cfg(epochs=3)
    a = phase4.run_head(emb, cfg, head_layers=0, seed=1, eval_splits=["val"])
    assert "test" not in a and "test" not in seen
    b = phase4.run_head(emb, cfg, head_layers=0, seed=1, eval_splits=["val", "test"])
    assert np.array_equal(a["val"], b["val"]) and len(b["test"]) == 15
    lr = phase4.run_lr(emb, 0.1, ["val"])
    assert set(lr) == {"val"} and len(lr["val"][1]) == 15
