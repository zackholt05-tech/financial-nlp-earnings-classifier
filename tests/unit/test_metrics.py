"""Metric behaviour on small, empty and single-class samples."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from earnings_classifier.baselines import logistic_regression_baseline, majority_class_baseline
from earnings_classifier.config import Config
from earnings_classifier.evaluate import compute_metrics, format_metrics
from earnings_classifier.model import FusionHead
from earnings_classifier.preprocessing import EmbeddingDataset
from earnings_classifier.train import frozen_forward, train_model


def test_empty_sample_is_all_undefined():
    m = compute_metrics([], [])
    assert m["n"] == 0 and math.isnan(m["auc"]) and math.isnan(m["accuracy"])
    assert m["undefined"] == {"all": "empty sample"}
    format_metrics("empty", m)  # must not raise


def test_single_example():
    m = compute_metrics([1], [0.8], train_pos_rate=0.4)
    assert math.isnan(m["auc"]) and math.isnan(m["average_precision"])
    assert m["accuracy"] == 1.0 and m["majority_class_acc"] == 0.0
    assert m["brier"] == pytest.approx(0.04)
    assert m["log_loss"] == pytest.approx(-math.log(0.8))


def test_single_class_sample_reports_reasons():
    m = compute_metrics([0, 0, 0], [0.2, 0.6, 0.1])
    assert math.isnan(m["auc"]) and m["undefined"]["auc"] == "single-class sample"
    assert math.isnan(m["recall"]) and "recall" in m["undefined"]
    assert m["precision"] == 0.0 and m["accuracy"] == pytest.approx(2 / 3)


def test_no_positive_predictions_precision_undefined():
    m = compute_metrics([0, 1, 1], [0.1, 0.2, 0.3])
    assert math.isnan(m["precision"]) and m["undefined"]["precision"] == "no positive predictions"
    assert m["recall"] == 0.0 and m["f1"] == 0.0


def test_perfect_and_inverted_rankings():
    y = [0, 0, 1, 1]
    assert compute_metrics(y, [0.1, 0.2, 0.8, 0.9])["auc"] == 1.0
    assert compute_metrics(y, [0.9, 0.8, 0.2, 0.1])["auc"] == 0.0
    assert compute_metrics(y, [0.1, 0.2, 0.8, 0.9])["average_precision"] == 1.0


def test_probability_metrics_hand_computed():
    y, p = [0, 1, 1, 0], [0.25, 0.5, 0.75, 0.5]
    m = compute_metrics(y, p)
    assert m["brier"] == pytest.approx((0.25**2 + 0.5**2 + 0.25**2 + 0.5**2) / 4)
    expected_ll = -(math.log(0.75) + math.log(0.5) + math.log(0.75) + math.log(0.5)) / 4
    assert m["log_loss"] == pytest.approx(expected_ll)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]   # threshold 0.5 -> 0.5 predicts up


def test_extreme_probabilities_have_finite_log_loss():
    assert math.isfinite(compute_metrics([1, 0], [0.0, 1.0])["log_loss"])


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.5])
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0.5, float("nan")])


def test_majority_baseline_uses_train_prevalence_not_eval_labels():
    m = majority_class_baseline(np.array([1, 1, 1, 0]), np.array([0, 0, 0, 1]))
    assert m["auc"] == 0.5
    assert m["accuracy"] == 0.25 and m["majority_class_acc"] == 0.25   # predicts "up"
    assert m["brier"] == pytest.approx(np.mean((0.75 - np.array([0, 0, 0, 1])) ** 2))


def test_lr_baseline_requires_two_training_classes():
    X = np.zeros((4, 3))
    with pytest.raises(ValueError, match="both classes"):
        logistic_regression_baseline(X, np.ones(4, int), X, np.array([0, 1, 0, 1]))


def _loader(labels):
    n = len(labels)
    ds = EmbeddingDataset(torch.randn(n, 4), torch.randn(n, 4), torch.ones(n), torch.ones(n),
                          torch.tensor(labels, dtype=torch.float32))
    return DataLoader(ds, batch_size=4)


def test_model_selection_refuses_single_class_validation():
    cfg = Config.tiny(epochs=1)
    with pytest.raises(ValueError, match="validation split"):
        train_model(FusionHead(cfg, embed_dim=4), _loader([0, 1, 0, 1]), _loader([1, 1, 1]),
                    np.array([0, 1, 0, 1]), cfg, torch.device("cpu"), frozen_forward)
