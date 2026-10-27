### Data and split

| split | entries | n | up | up-rate |
|---|---|---|---|---|
| train | 2016-10-19 .. 2023-04-24 | 740 | 348 | 0.470 |
| val | 2023-04-27 .. 2024-10-24 | 169 | 89 | 0.527 |
| test | 2024-10-31 .. 2026-10-23 | 229 | 104 | 0.454 |

Purged (label window overlaps the next split): train 6, val 4.

### Validation grid (selection)

| model | chunks | setting | val AUC |
|---|---|---|---|
| Logistic regression | 1 | C = 0.01 | 0.539 |
| Logistic regression | 1 | C = 0.1 | 0.523 |
| Logistic regression | 1 | C = 1.0 | 0.524 |
| Logistic regression | 8 | C = 0.01 | 0.498 |
| Logistic regression | 8 | C = 0.1 | 0.453 |
| Logistic regression | 8 | C = 1.0 | 0.452 |
| Fusion head | 1 | 0 hidden layers | 0.602 ± 0.008 (5 seeds) |
| Fusion head | 1 | 1 hidden layers | 0.594 ± 0.005 (5 seeds) |
| Fusion head | 8 | 0 hidden layers | 0.568 ± 0.010 (5 seeds) |
| Fusion head | 8 | 1 hidden layers | 0.579 ± 0.007 (5 seeds) |

Selected: logistic regression {'max_document_chunks': 1, 'C': 0.01}, fusion head {'max_document_chunks': 1, 'head_layers': 0}.

### Held-out test (evaluated once)

229 events in 79 entry weeks; 95% intervals from 2000 week-block bootstrap resamples.

| model | val AUC | test AUC | 95% CI | AP | Brier | log loss | accuracy | F1 |
|---|---|---|---|---|---|---|---|---|
| Train-majority | 0.500 | 0.500 | [0.500, 0.500] | 0.454 | 0.248 | 0.689 | 0.546 | 0.000 |
| Ticker prior | 0.566 | 0.487 | [0.423, 0.556] | 0.448 | 0.258 | 0.710 | 0.528 | 0.407 |
| Logistic regression | 0.539 | 0.461 | [0.383, 0.538] | 0.456 | 0.293 | 0.786 | 0.450 | 0.515 |
| FinBERT fusion head | 0.601 | 0.506 | [0.439, 0.573] | 0.455 | 0.259 | 0.711 | 0.463 | 0.580 |
| Zero-shot FinBERT sentiment | 0.532 | 0.469 | [0.388, 0.552] | 0.434 | — | — | — | — |

| paired difference (test AUC) | estimate | 95% CI | share of resamples ≤ 0 |
|---|---|---|---|
| FinBERT fusion head − Ticker prior | +0.019 | [-0.052, +0.089] | 0.31 |
| Logistic regression − Ticker prior | -0.026 | [-0.111, +0.052] | 0.73 |
| Zero-shot FinBERT sentiment − Ticker prior | -0.018 | [-0.119, +0.081] | 0.66 |
| FinBERT fusion head − Logistic regression | +0.045 | [-0.042, +0.129] | 0.16 |
| Ticker prior − Train-majority | -0.013 | [-0.077, +0.056] | 0.64 |

| fusion-head seed | best epoch | val AUC | test AUC |
|---|---|---|---|
| 0 | 3 | 0.617 | 0.506 |
| 1 | 10 | 0.600 | 0.507 |
| 2 | 3 | 0.605 | 0.497 |
| 3 | 16 | 0.593 | 0.503 |
| 4 | 26 | 0.596 | 0.516 |

### Exploratory: headlines (no training, no selection)

111 events with pre-cutoff headlines, entries 2025-11-05 .. 2026-10-23, up-rate 0.468.

| sample | score | n | AUC | 95% CI |
|---|---|---|---|---|
| all with headlines | headline sentiment | 111 | 0.429 | [0.323, 0.556] |
| headlines and release | headline sentiment | 109 | 0.435 | [0.329, 0.563] |
| headlines and release | release sentiment | 109 | 0.413 | [0.305, 0.536] |

Headline − release sentiment AUC: +0.022 [-0.127, +0.182].

### Synthetic pipeline controls (FinBERT, seeds 0–4; not market evidence)

| control | runs | neural test AUC (mean ± sd) | range | LR test AUC (mean) |
|---|---|---|---|---|
| synthetic-negative-control | 5 | 0.465 ± 0.075 | 0.392 .. 0.578 | 0.469 |
| synthetic-positive-control | 5 | 1.000 ± 0.000 | 1.000 .. 1.000 | 1.000 |
