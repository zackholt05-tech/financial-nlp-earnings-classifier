"""Run directories, manifests and secret hygiene."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

import pytest

from earnings_classifier.config import NEGATIVE_CONTROL
from earnings_classifier.provenance import create_run_dir, write_json

FAKE_KEY = "fake-key-for-tests-987"   # not a credential; checks that env values never leak
REQUIRED = {"manifest_version", "created_at", "data_mode", "is_control", "argv", "seed",
            "config", "dataset_fingerprint", "events_cache", "embeddings_cache", "model",
            "counts", "dependencies", "platform", "git"}


def test_run_dir_is_never_reused(tmp_path):
    now = datetime(2026, 10, 27, 12, 0, 0, 123000, tzinfo=timezone.utc)
    path = create_run_dir(str(tmp_path), NEGATIVE_CONTROL, "ab" * 32, now=now)
    assert os.path.basename(path) == f"20261027T120000123Z_{NEGATIVE_CONTROL}_abababab"
    with pytest.raises(FileExistsError):
        create_run_dir(str(tmp_path), NEGATIVE_CONTROL, "ab" * 32, now=now)


def test_artifacts_are_never_overwritten(tmp_path):
    write_json(str(tmp_path / "m.json"), {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(str(tmp_path / "m.json"), {"a": 2})


def test_full_run_manifest_and_no_key_leak(tmp_path, monkeypatch, fake_models):
    import main as main_mod

    monkeypatch.setenv("FMP_API_KEY", FAKE_KEY)
    monkeypatch.setenv("FINNHUB_API_KEY", FAKE_KEY)
    monkeypatch.chdir(tmp_path)
    run_dir = main_mod.main(["--data-mode", NEGATIVE_CONTROL, "--preset", "tiny",
                             "--n-synthetic", "60", "--epochs", "2", "--device", "cpu",
                             "--runs-dir", str(tmp_path / "runs"), "--seed", "3"])
    assert sorted(os.listdir(run_dir)) == ["head.pt", "manifest.json", "metrics.json",
                                           "predictions.csv"]
    m = json.load(open(os.path.join(run_dir, "manifest.json")))
    assert REQUIRED <= set(m)
    assert m["data_mode"] == NEGATIVE_CONTROL and m["is_control"] is True and m["seed"] == 3
    assert m["model"]["revision"] == m["config"]["model_revision"]
    assert len(m["model"]["revision"]) == 40
    splits = m["counts"]["split"]["splits"]
    assert sum(s["n"] for s in splits.values()) == m["counts"]["used"]
    assert {"torch", "transformers", "numpy", "scikit-learn", "python"} <= set(m["dependencies"])
    assert os.path.basename(run_dir).endswith(m["dataset_fingerprint"][:8])

    metrics = json.load(open(os.path.join(run_dir, "metrics.json")))
    assert set(metrics["metrics"]) == {"neural", "logistic_regression", "majority_class"}
    rows = list(csv.DictReader(open(os.path.join(run_dir, "predictions.csv"))))
    assert {r["split"] for r in rows} == {"val", "test"}
    assert len(rows) == splits["val"]["n"] + splits["test"]["n"]

    for root, _, files in os.walk(tmp_path):
        for f in files:
            with open(os.path.join(root, f), "rb") as fh:
                assert FAKE_KEY.encode() not in fh.read(), f"env value leaked into {f}"


def test_manifests_record_repo_relative_paths(tmp_path):
    from earnings_classifier.provenance import build_manifest

    root = str(tmp_path / "repo")
    m = build_manifest(root=root, argv=["main.py", "--runs-dir", f"{root}/runs/x"],
                       config={"runs_dir": f"{root}/runs/x", "other": "/elsewhere/y", "n": 3})
    assert m["argv"] == ["main.py", "--runs-dir", "runs/x"]
    assert m["config"] == {"runs_dir": "runs/x", "other": "/elsewhere/y", "n": 3}

