"""Uncertainty estimates that respect temporal dependence.

Events that enter in the same week share market-wide moves, so their labels are
not independent draws. The week-block bootstrap resamples whole entry weeks
(with replacement, as many weeks as the sample has), recomputes every model's
ROC AUC on the same resample, and reports percentile intervals for each AUC
and for paired AUC differences. Resamples that contain a single class (AUC
undefined) are skipped and counted.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def entry_weeks(entry_dates: Sequence[str]) -> np.ndarray:
    """ISO year-week label ("2024-W45") of each ISO entry date."""
    out = []
    for d in entry_dates:
        y, w, _ = date.fromisoformat(str(d)).isocalendar()
        out.append(f"{y}-W{w:02d}")
    return np.array(out)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _week_index(groups: np.ndarray) -> List[np.ndarray]:
    return [np.flatnonzero(groups == g) for g in np.unique(groups)]


def draw_block_indices(blocks: List[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Event indices of one resample: len(blocks) whole blocks drawn with replacement."""
    pick = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[i] for i in pick])


def week_block_bootstrap(
    y: Sequence[int],
    scores: Dict[str, Sequence[float]],
    weeks: Sequence[str],
    pairs: Optional[List[Tuple[str, str]]] = None,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Dict:
    """Point AUCs, percentile intervals, and paired differences `a - b` for `pairs`."""
    y = np.asarray(y).astype(int)
    weeks = np.asarray(weeks)
    S = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    for k, v in S.items():
        if v.shape != y.shape or not np.isfinite(v).all():
            raise ValueError(f"scores[{k!r}] must be finite and match y")
    if weeks.shape != y.shape:
        raise ValueError("weeks must match y")
    pairs = list(pairs or [])
    blocks = _week_index(weeks)
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in S}
    dboot = {f"{a} - {b}": [] for a, b in pairs}
    skipped = 0
    for _ in range(n_boot):
        idx = draw_block_indices(blocks, rng)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            skipped += 1
            continue
        aucs = {k: _auc(yb, v[idx]) for k, v in S.items()}
        for k, a in aucs.items():
            boot[k].append(a)
        for a, b in pairs:
            dboot[f"{a} - {b}"].append(aucs[a] - aucs[b])

    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2

    def interval(vals: List[float], point: float) -> Dict:
        v = np.asarray(vals)
        return {"point": point, "lo": float(np.percentile(v, lo_q)),
                "hi": float(np.percentile(v, hi_q)), "boot_sd": float(v.std(ddof=1))}

    point = {k: _auc(y, v) for k, v in S.items()}
    out = {
        "n_events": int(len(y)), "n_weeks": len(blocks), "n_boot": n_boot,
        "n_skipped_single_class": skipped, "level": level, "seed": seed,
        "auc": {k: interval(boot[k], point[k]) for k in S},
        "diff": {},
    }
    for a, b in pairs:
        name = f"{a} - {b}"
        d = interval(dboot[name], point[a] - point[b])
        d["frac_le_0"] = float(np.mean(np.asarray(dboot[name]) <= 0))
        out["diff"][name] = d
    return out
