# Implementation plan (phases 1–3)

Goal: make the existing FinBERT fusion pipeline produce experiments that are trustworthy and reproducible, before any real-data benchmarking. The live progress log is in [STATUS.md](STATUS.md).

## Phase 1: Audit and preserve
- Copy the pre-audit `artifacts/` and `data/cache/` into `runs/legacy-2026-06-18-positive-control/` (checksummed, with a provenance note). Leave the originals untouched.
- `git init` with private files ignored, then make a baseline commit of the unmodified source.
- Run the unmodified smoke test, reproduce the cache bugs end-to-end on scratch copies, and probe the providers ([DATA_SOURCES.md](DATA_SOURCES.md)).

## Phase 2: Experiment correctness (regression tests first)
1. **Data modes.**
   - `--data-mode {real, synthetic-negative-control, synthetic-positive-control}`.
   - Real mode never falls back to synthetic data. It raises `RealDataError` with a coverage report when there are no text providers, collection fails, or too few usable events remain.
2. **Information timeline.** All times are in America/New_York. For each event:
   - **Announcement time `A`** and session (BMO before 09:30, DMH, AMC at or after 16:00, UNKNOWN when the time is missing or midnight). Taken from yfinance.
   - **Earnings document.** By default the SEC EDGAR earnings press release (8-K Item 2.02, Exhibit 99.1), dated by its exact acceptance time in ET; alternatively an FMP call transcript, dated by its call date with the provider's own fiscal year and quarter. Matched by *date proximity* to `A` within [-1, +3] days. The calendar quarter of `A` is never used. A document is never reused across events.
   - **News cutoff = 00:00 ET on the announcement date**, or the document's acceptance time if that is earlier. Only headlines published on earlier calendar days (and before the release) count. Provider announcement times are approximate (Yahoo stamps after-close reports at 16:00 and pre-open reports at 06:00/07:00), so same-day news is excluded rather than trusting the stamped minute. Headline timestamps are stored and re-checked when features are assembled.
   - **Information date `D = max(date(A), document date)`.**
   - **Prediction time.** The 09:30 ET open of the first trading day strictly after `D`, the *entry day*. This is the "post-call entry" definition chosen for the research question. It deliberately skips a BMO same-day open, because the call may still be running at 09:30. As a result, it does not depend on imprecise announcement times.
   - **Label.** `sign(Close[exit] / Open[entry] - 1)`, where `exit` is the `horizon_days`-th trading session counting the entry day as session 1. Trading days come from the price index itself, so holidays are handled automatically.
3. **Split protocol.**
   - Order events chronologically by entry day and cut only at date boundaries, so same-date groups stay together.
   - Boundaries come from fractions or explicit dates.
   - Purge any train event whose label window ends on or after the validation start, and any validation event whose window ends on or after the test start.
   - Splits are recomputed on every run and never cached.
4. **Caches.**
   - **Event cache.** Keyed on the schema version, the data mode and every collection-relevant setting. Provenance is stored in the file and verified on load.
   - **Embedding cache.** Keyed on the ordered event identities, a per-event content hash, labels, splits, the model and tokenizer revision (pinned commit hashes), and the tokenization and pooling settings.
   - Legacy caches are never auto-loaded.
   - `--no-cache` disables cache reads and writes everywhere.
5. **Run directories.**
   - Each run writes to `runs/<UTC timestamp>_<mode>_<fingerprint>/` and refuses to overwrite an existing directory.
   - Contents: `manifest.json` (config, seed, argv, mode, fingerprints, cache keys, model identity, counts, class balance, date ranges, coverage, dependency versions, git commit), `metrics.json`, `predictions.csv`, and local-only weights.
6. **Evaluation.**
   - Add average precision, Brier score and log loss.
   - Inputs that are empty or single-class return explicit "undefined" reasons.
   - The majority-class baseline uses the train-split majority.
   - Model selection refuses a single-class validation split.

## Phase 3: Test workflow
- `pyproject.toml` with pytest config. Network-free unit tests use a fake tokenizer and encoder, mocked providers and hand-made fixtures.
- The tiny-BERT end-to-end smoke test is marked `slow`.
- A tiny-encoder demo for both controls.
- A dependency lock file, verified in a clean virtual environment with an isolated `HF_HOME`.
- A GitHub Actions workflow. Its status stays "pending first push" until the repository exists.

## Decisions
- **Primary label.** Post-call entry, as described above; chosen on 2026-10-27. No alternative "announcement reaction" label is implemented.
- **API keys.** Free-tier keys only. Free-tier limits, such as about one year of Finnhub news, will be measured and reported, not assumed away.
- **Document stream (2026-10-27).** SEC EDGAR earnings press releases are the default real-data document, chosen after FMP transcripts proved paid-only. The paper describes press-release text, not call transcripts.
