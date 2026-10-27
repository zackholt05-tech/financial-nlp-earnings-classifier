# Phase 4 protocol (APPROVED 2026-10-27)

Written 2026-10-27, after the coverage dry run and **before any model has been trained on real data**. It is frozen: any later change is recorded in the change log at the end with a date and a reason, and test results are never used to choose between alternatives.

**Approval.** The open decisions were resolved on 2026-10-27, toward robustness, and recorded in §7 before any real-data model was trained.

## 1. Data (frozen)
- **Primary dataset:** document-only real events, collected once with `--headlines none`.
  - 30 tickers × 40 most recent announcements; SEC EDGAR earnings press releases (8-K Item 2.02).
  - **1,148 events.** Fingerprint `2caa9d48aedd07ab…` (full value in the cache file); cache `data/cache/events-real-88412e2753d3aca7.json`.
  - Every experiment reads this cache. It is never re-collected: Yahoo responses are not raw-cached, and the provider data keeps moving.
- **Why document-only.** Free Finnhub news covers only about the last 12 months, so headlines exist **only in the test period**: train 0 of 740 events, val 0 of 169.
  - A fusion head trained without headlines would see test-time headline features it never learned. Its interaction terms (`|t−n|`, `t⊙n`) and the presence weights would be fit on zeros.
  - Masking headlines on the mixed dataset is not equivalent. Its two headline-only events would stay in as all-zero rows.
- **Headlines.** Coverage counts are reported: 111 test events have headlines, 109 of them alongside a release. Headlines enter only the exploratory, training-free analysis in §7.

## 2. Split (frozen before model selection)
Chronological by entry date. The cuts fall at date boundaries, and label windows overlapping the next split are purged. The 65/15/20 fractions give the following boundaries, which are pinned as explicit config values (`val_start_date`, `test_start_date`):

| split | entries | n | up-rate |
|---|---|---|---|
| train | 2016-10-19 .. 2023-04-24 | 740 | 0.470 |
| val | 2023-04-27 .. 2024-10-24 | 169 | 0.527 |
| test | 2024-10-31 .. 2026-10-23 | 229 | 0.454 |

Purged: 6 train events and 4 val events. Up-rates differ across splits, so accuracy versus the train-majority baseline will look poor or good by regime alone. **ROC AUC is the primary metric.**

## 3. Models compared
1. **Train-majority baseline.**
2. **Ticker-prior baseline.** Each ticker's training up-rate, shrunk toward the global training rate with weight 10 (`(ups + 10·global) / (n + 10)`), used as the score.
   - This is the reason the baseline exists: every release names its company, so the embedding encodes ticker identity, and a model can beat 0.5 just by recognising companies that kept rising.
3. **Logistic regression** on the frozen FinBERT release embedding. Selected on val over `max_document_chunks ∈ {1, 8}` × `C ∈ {0.01, 0.1, 1}`.
4. **FinBERT fusion head** (frozen encoder; only the document stream is present). Selected on val over `max_document_chunks ∈ {1, 8}` × `head_layers ∈ {0, 1}`.
   - One chunk is the headline and summary paragraph; eight chunks is ~3k tokens.
   - Every grid point is trained with 5 seeds (0–4), and the configuration with the best mean val AUC wins. Ties go to the simpler configuration.
   - Embeddings are computed once on the available accelerator; the heads train on CPU so that seeds reproduce exactly.
5. **Zero-shot FinBERT sentiment** (no training, no selection). The pinned FinBERT checkpoint's own classifier (labels positive / negative / neutral, read from its config) scores each of the same 8 release chunks; the event score is the mean over chunks of `P(positive) − P(negative)`. It is rank-only, so only AUC and AP are reported for it.

**Interpretation rule (fixed now).** "Release-language signal" may be claimed only if a FinBERT model beats the **ticker prior** on the paired week-bootstrap AUC difference (95% interval excluding 0). Beating 0.5 alone is not enough.

**Not included:** end-to-end fine-tuning (740 training events, no GPU, high overfitting risk) and headline-versus-document ablations (§1).

## 4. Before the first real training run (requirements)
- Copy the frozen events file aside (`data/frozen/`) and record its sha256 in STATUS. `--refresh-cache` would otherwise overwrite it, and Yahoo responses cannot be re-fetched identically.
- Add `--expect-fingerprint <sha256>`. Real runs abort on a fingerprint mismatch or an events-cache miss, so a forgotten `--headlines none` or a collector-version bump can never silently re-collect from live providers.
- Add `--val-start-date` / `--test-start-date` to the CLI. Then confirm with `--collect-only` that the explicit dates reproduce 740 / 169 / 229 with 6 + 4 purged.
- Implement the ticker-prior baseline, the LR grid, the week-block bootstrap and the paired differences, each with unit tests.

## 5. Final test
The selected configuration of each model is frozen in a committed file **before** test metrics are computed. It is then evaluated **once**: 5 seeds, mean and range reported. No change follows from test results.

## 6. Metrics and uncertainty
- **Metrics** on val and test: n, date range, class balance, ROC AUC, average precision, accuracy, F1, Brier score, log loss.
- **Bootstrap input.** Neural results use seed-averaged probabilities per event; per-seed spread is reported separately.
- **95% intervals:** a block bootstrap that resamples **entry weeks** (2,000 draws). Events in the same week share market moves, so they are not treated as independent.
- **Paired comparisons:** bootstrap differences in AUC between models, using the same week resamples.
- **Expected precision.** Test spans about 8 earnings seasons, so it holds only a few dozen effective week clusters. Intervals will be wide; this is stated up front, not discovered.
- **Reporting:** a null result (AUC ≈ 0.5) is reported as such. Classification metrics are not trading performance.

## 7. Decisions (resolved 2026-10-27)
1. **Primary benchmark:** the document-only dataset and the frozen split above.
2. **Headlines: exploratory and training-free only.**
   - Dataset: the frozen with-headlines collection (1,150 events; fingerprint `ce118a67e773bc89…`), restricted to events with at least one pre-cutoff headline (111 events, all from the last ~12 months).
   - Scores: zero-shot FinBERT sentiment of the headlines (mean over the up to 16 most recent headlines of `P(positive) − P(negative)`), and the same zero-shot score of the press release for the same events.
   - Reported: AUC with week-block bootstrap intervals, labelled **exploratory** everywhere. Nothing is trained or selected on these events, and they never feed the primary benchmark.
3. **Controls:** both synthetic controls are re-run with FinBERT, seeds 0–4 each, through `main.py` (400 events per seed), and labelled as pipeline checks.
4. **Not run:** end-to-end fine-tuning (no GPU was used).

## 8. Known limitations for the paper
- **Survivorship bias.** The universe is today's 30 large caps projected back to 2016.
- **Coverage gaps.** XOM is absent from EDGAR in this form, DIS has no documents before 2019, and 4 events have no 8-K near the Yahoo date.
- **Truncation.** About 3k tokens are encoded from releases with a median length of ~29k characters.
- **Heterogeneous formats.** Some releases are shareholder letters or decks (NFLX, TSLA).
- **Yahoo timing.** Announcement times are approximate. The conservative entry rule makes results insensitive to the exact minute.
- **Reproduction.** The frozen events file contains provider data and is not redistributed. Live re-collection drifts (rolling news history, revised Yahoo data), so an outside reproduction builds a different dataset; only the code and the dataset fingerprint are shareable.
- **Ticker identity.** Releases name the company, so embeddings carry ticker identity (see the ticker-prior baseline).

## Change log
- 2026-10-27: drafted; decisions resolved as in §7 and the zero-shot FinBERT baseline added (§3.5), all before any real-data training.
