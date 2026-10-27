"""Reference baselines.

* Majority class: predicts the training-split prevalence for every event
  (AUC 0.5 by construction; its Brier/log loss are the "know nothing" bar).
* Logistic regression on the frozen [document | news | presence] features:
  if the neural fusion head can't beat it, the extra capacity is just
  memorizing. With only hundreds of events, a strong linear baseline is often
  the most trustworthy number in the whole report.
* Ticker prior: each company's training up-rate (shrunk toward the global
  rate). The bar a text model must clear before "language signal" is claimed.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import compute_metrics


def majority_class_baseline(y_train: np.ndarray, y_eval: np.ndarray) -> Dict:
    if len(y_train) == 0:
        raise ValueError("majority baseline needs a non-empty training split")
    rate = float(np.mean(y_train))
    return compute_metrics(y_eval, np.full(len(y_eval), rate), train_pos_rate=rate)


def logistic_regression_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int = 42,
    C: float = 0.5,
) -> Dict:
    """Returns (metrics, eval probabilities)."""
    if len(np.unique(y_train)) < 2:
        raise ValueError("logistic-regression baseline needs both classes in the training split")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=C, class_weight="balanced", random_state=seed),
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_eval)[:, 1]
    return compute_metrics(y_eval, prob, train_pos_rate=float(np.mean(y_train))), prob


def ticker_prior_baseline(train_tickers: Sequence[str], y_train: np.ndarray,
                          eval_tickers: Sequence[str], y_eval: np.ndarray,
                          shrinkage: float = 10.0) -> Tuple[Dict, np.ndarray]:
    """Score = the ticker's training up-rate, shrunk toward the global training rate:
    (ups + shrinkage * global) / (n + shrinkage). Unseen tickers get the global rate.

    It uses only *which company* an event belongs to, so a text model that cannot
    beat it has not shown that the language itself carries signal (the documents
    name their company, so embeddings encode identity). Returns (metrics, probs)."""
    y_train = np.asarray(y_train).astype(int)
    if len(y_train) == 0:
        raise ValueError("ticker prior needs a non-empty training split")
    glob = float(y_train.mean())
    ups: Dict[str, int] = {}
    cnt: Dict[str, int] = {}
    for t, y in zip(train_tickers, y_train):
        ups[t] = ups.get(t, 0) + int(y)
        cnt[t] = cnt.get(t, 0) + 1
    prob = np.array([(ups.get(t, 0) + shrinkage * glob) / (cnt.get(t, 0) + shrinkage)
                     for t in eval_tickers], dtype=float)
    return compute_metrics(y_eval, prob, train_pos_rate=glob), prob
