"""Evaluation metrics.

ROC AUC is the primary ranking metric; average precision, Brier score and log
loss describe ranking-at-the-top and probability quality. Accuracy is reported
next to the majority-class rule *learned on the training split* (not the
evaluation split's own majority, which would peek at the evaluation labels).

Metrics that are undefined for the given sample (empty input, a single class,
no positive predictions) are NaN and the reason is recorded in `undefined`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

NAN = float("nan")
_EPS = 1e-15


def compute_metrics(y_true, y_prob, threshold: float = 0.5,
                    train_pos_rate: Optional[float] = None) -> Dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true {y_true.shape} and y_prob {y_prob.shape} differ in shape")
    if y_prob.size and (not np.isfinite(y_prob).all() or y_prob.min() < 0 or y_prob.max() > 1):
        raise ValueError("y_prob must be finite probabilities in [0, 1]")
    n = int(y_true.size)
    undefined: Dict[str, str] = {}
    out: Dict = {"n": n, "n_pos": int(y_true.sum()), "n_neg": int(n - y_true.sum())}

    if n == 0:
        for k in ("auc", "average_precision", "brier", "log_loss", "accuracy", "f1",
                  "precision", "recall", "pos_rate", "majority_class_acc"):
            out[k] = NAN
        out["confusion_matrix"] = [[0, 0], [0, 0]]
        out["undefined"] = {"all": "empty sample"}
        return out

    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    if len(np.unique(y_true)) < 2:
        out["auc"] = out["average_precision"] = NAN
        undefined["auc"] = undefined["average_precision"] = "single-class sample"
    else:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["average_precision"] = float(average_precision_score(y_true, y_prob))

    p = np.clip(y_prob, _EPS, 1 - _EPS)
    out["brier"] = float(np.mean((y_prob - y_true) ** 2))
    out["log_loss"] = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    out["accuracy"] = float(np.mean(y_pred == y_true))
    if tp + fp == 0:
        out["precision"] = NAN
        undefined["precision"] = "no positive predictions"
    else:
        out["precision"] = tp / (tp + fp)
    if tp + fn == 0:
        out["recall"] = NAN
        undefined["recall"] = "no positive labels"
    else:
        out["recall"] = tp / (tp + fn)
    out["f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else NAN
    if np.isnan(out["f1"]):
        undefined["f1"] = "no positive labels or predictions"
    out["pos_rate"] = float(y_true.mean())
    if train_pos_rate is None:
        out["majority_class_acc"] = NAN
        undefined["majority_class_acc"] = "train prevalence not supplied"
    else:
        majority = int(train_pos_rate >= 0.5)
        out["majority_class_acc"] = float(np.mean(y_true == majority))
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    out["undefined"] = undefined
    return out


def _fmt(x: float) -> str:
    return "  n/a" if x != x else f"{x:.3f}"


def format_metrics(name: str, m: Dict) -> str:
    lift = ""
    if m["majority_class_acc"] == m["majority_class_acc"]:
        delta = m["accuracy"] - m["majority_class_acc"]
        lift = f" (train-majority {m['majority_class_acc']:.3f}, {delta:+.3f})"
    return (
        f"[{name}]  n={m['n']}  AUC={_fmt(m['auc'])}  AP={_fmt(m['average_precision'])}  "
        f"Brier={_fmt(m['brier'])}  LogLoss={_fmt(m['log_loss'])}\n"
        f"        acc={_fmt(m['accuracy'])}{lift}  F1={_fmt(m['f1'])}  "
        f"P={_fmt(m['precision'])}  R={_fmt(m['recall'])}  pos_rate={_fmt(m['pos_rate'])}  "
        f"confusion [[TN,FP],[FN,TP]] = {m['confusion_matrix']}"
    )
