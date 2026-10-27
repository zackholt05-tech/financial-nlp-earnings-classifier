# Phase 4 results

Real-data benchmark run exactly as pre-registered in [PHASE4_PROTOCOL.md](PHASE4_PROTOCOL.md), which was committed before any real-data training. Selection used validation data only and was committed before the test set was evaluated, once. The public repository starts from a single commit, so this order is documented in the protocol's change log and in the development-history commit identifiers stored in the result files, not in the public git history, and it was not lodged with an external registry. All tables below are generated from the result files by `experiments/report_tables.py`; raw results are in [`results/`](../results/).

## Summary
- **Pipeline checks pass.** With FinBERT, the negative control (labels independent of text) scores test AUC 0.465 ± 0.075 over 5 seeds, consistent with chance at n ≈ 80 per seed. The positive control (text encodes the label) scores 1.000 on every seed. Signal that is present in the text reaches the classifier, and labels do not leak into the features.
- **Real data: a null result.** On 229 held-out events (2024-10-31 to 2026-10-23), no model's test AUC is distinguishable from 0.5. The best, the FinBERT fusion head, scores 0.506 (95% week-block CI 0.439–0.573).
- **Interpretation rule.** No model beats the ticker-prior baseline on the paired bootstrap (fusion head − ticker prior: +0.019, CI −0.052 to +0.089), so, by the rule fixed in advance, **no release-language signal is claimed**.
- **Validation was optimistic.** Validation AUCs of 0.54–0.60 did not carry over to test.
  - Part of that gap is built in: the fusion head's validation AUC is the epoch chosen by early stopping on that same validation set.
  - The ticker prior also fell from 0.566 on validation to 0.487 on test, a sign that which companies kept rising changed between periods.
- **Headlines (exploratory) are null too.** Zero-shot FinBERT sentiment of pre-announcement headlines scores AUC 0.429 (CI 0.323–0.556) on the 111 events of the last year that have headlines. Release sentiment on the same events is no different (+0.022, CI −0.127 to +0.182).
- **What the label measures.** The label is the post-announcement drift, not the reaction. Entry is the open *after* the information date, so the immediate response is already in the entry price:
  - after-close reports: the overnight gap;
  - pre-open reports: the whole announcement day.

  Mega-cap prices absorb a release quickly, so a null on drift is the expected outcome. It is not evidence that release text cannot explain the announcement reaction, which was not tested.
- **This is not trading performance.** None of these numbers measures one, and none supports a trading claim.

## What the null result does and does not say
- **It says:** for 30 large US tickers on this split (XOM contributes almost no events; see DATA_SOURCES), these do not rank the sign of the next three sessions' open-to-close return better than chance:
  - frozen-FinBERT representations of the first ~380 tokens of a company's earnings press release (the selected setting; the ~3k-token setting was only evaluated on validation, where it did no better);
  - FinBERT's own zero-shot sentiment of the first ~3k tokens.
- **It does not test:**
  - the immediate announcement reaction, which is already in the entry price;
  - call transcripts or analyst Q&A;
  - fine-tuned encoders;
  - text that arrives before the announcement;
  - the surprise against expectations;
  - smaller or less-covered companies, where text might matter more;
  - longer or shorter horizons.
- **Precision.** With 229 test events in 79 entry weeks, the 95% interval on AUC is about ±0.07. Effects smaller than that cannot be distinguished from zero here.
- **Consistency with priors.** The result matches the expectation stated before any run (README, "Expect ~0.5 AUC from a leak-free model on real data").

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

## Notes on the table
- **Majority baseline.** The train-majority score is the training up-rate (0.470) for every event. Its AUC is 0.5 by construction, and it predicts "down" everywhere, hence F1 = 0. Its Brier score (0.248) and log loss (0.689) are the "knows nothing" bar. No model beats them on test.
- **Zero-shot FinBERT.** Its score is `P(positive) − P(negative)`, which is rank-only, so probability metrics are not reported for it.
- **Bootstrap.** The bootstrap uses seed-averaged probabilities for the fusion head. The per-seed table shows the spread across seeds: test AUC 0.497–0.516.
- **Exploratory headlines.** The events are a subset of the primary test period. Nothing was trained or selected on them, and the 39 entry weeks make the intervals wide.

## Reproduction
```bash
python experiments/phase4.py select      # validation only; writes results/phase4/selection.json
python experiments/phase4.py test        # refuses unless selection.json is committed; runs once
python experiments/phase4.py headlines   # exploratory
python experiments/summarize_controls.py runs/controls-finbert
python experiments/report_tables.py      # these tables
```
The frozen events files are not redistributed (provider data). Their fingerprints and sha256 hashes are recorded in [STATUS.md](STATUS.md), rows 34–35. A fresh collection with the same code builds a similar but not identical dataset, because provider data changes over time.
