# Status and verification log

The plan is in [PLAN.md](PLAN.md). Every check listed here was actually run; the exact command and its outcome are recorded. Commit identifiers in this log refer to the development history; the public repository starts from a single commit.

## Progress
- [x] Phase 1: audit, preservation, baseline commit, provider probes
- [x] Phase 2: experiment correctness. Data modes, information timeline, fiscal-period matching, date-grouped and purged splits, provenance-keyed caches, `--no-cache` / `--refresh-cache`, run directories and manifests, metric fixes
- [x] Phase 3: pytest config, 111 network-free unit tests plus a tiny-BERT smoke test, a tiny-encoder demo, a dependency lock verified in a clean clone, and a CI workflow. CI runs on every push.
- [x] Real-data document stream: SEC EDGAR earnings press releases (8-K Item 2.02, Exhibit 99.1) added as the default document provider, with the schema generalized from "transcript" to "document" (FMP transcripts remain selectable). Verified live (row 30)
- [x] Phase 4: pre-registered real-data benchmark (selection on validation, committed, test evaluated once), FinBERT controls ×5 seeds, exploratory zero-shot headline analysis. **Result: null** (see [RESULTS.md](RESULTS.md))
- [x] Phase 5: paper ([paper/main.pdf](../paper/main.pdf)): LaTeX source, verified bibliography, and every number, table and figure generated from `results/`; compiled with tectonic and inspected
- [x] Phase 6: README rewrite, `.env.example`, repo-relative manifest paths, sanitized control manifests, CI check that generated paper assets match the results
- [x] Phase 7: published (see the README for the repository URL); CI passes

## Decisions
- **2026-10-27: License.** MIT for the code (`LICENSE`, `pyproject.toml` `license = "MIT"`).
- **2026-10-27: API keys.** Free-tier keys are stored in the gitignored, `chmod 600` `.env`. They are loaded into each command's environment only and never printed.
- **2026-10-27: Both conservative timeline changes adopted.**
  - **Entry.** Every event, including pre-open (BMO) reports, enters at the open of the first trading day *after* the information date. This replaces the same-day BMO open in the original plan.
  - **News cutoff.** Headlines must be published before 00:00 ET on the announcement date.
- **2026-10-27: Phase 4 decisions.** Resolved toward robustness and recorded in the protocol before any real-data training: document-only primary benchmark on the frozen split; headlines only as an exploratory, training-free zero-shot analysis; FinBERT controls with 5 seeds; a zero-shot FinBERT sentiment baseline added.
- **2026-10-27: Real-data document stream = SEC EDGAR earnings press releases.** Free tiers gave headlines for about a year only, and FMP transcripts need a paid plan.
  - The paper's framing changes from call transcripts to press releases.
  - The contact email for the SEC User-Agent header lives only in the gitignored `.env` as `SEC_EDGAR_USER_AGENT` and is never written to the repository, caches or run artifacts.
  - One timeline rule follows (stricter, not looser): when the 8-K was accepted before 00:00 ET of the announcement date, the news cutoff moves back to the acceptance time, so no headline published after the release enters the features.

## Audit findings
These were found by reading every source file, the caches and the environment on 2026-10-27.

**Reproduced cache bugs** (end-to-end, on a scratch copy of the code plus the legacy caches; see the log below):
1. `collect_events()` loads `data/cache/events.json` before checking the requested data mode. A real-data request (`main.py` with no flags) returned 400 synthetic events.
2. `compute_embeddings()` accepts any cache with the same model name and event count. The legacy `events.json` is negative-control data, while the legacy `embeddings.npz` holds positive-control labels and embeddings. Consequences:
   - `main.py --synthetic` (the negative control) reported test AUC 1.000.
   - The banner still said "AUC ~0.5 is EXPECTED".
3. `--no-cache` bypasses only the event cache. `main.py --synthetic --no-cache` regenerated the events but still loaded the stale positive-control embeddings: test AUC 1.000.

**Other correctness problems** that affect the planned experiments:
- **Real-mode fallbacks.**
  - Real mode silently falls back to synthetic data when there are fewer than `min_real_events` events.
  - With no keys set, it keeps events with empty text.
  - Provider errors are swallowed at debug or warning level.
- **Inconsistent label anchor.** The label starts at close[first trading day ≥ announcement date].
  - For after-close (AMC) reports that close is before the news.
  - For pre-open (BMO) reports it is after the news.
- **UTC date roll.** ET timestamps are converted to naive UTC before `.date()`, so late-evening ET timestamps roll to the next date. There is no session classification.
- **Transcript quarter.** Computed as the calendar quarter of the announcement month. This is wrong for non-calendar fiscal years and for the usual one-quarter reporting lag. A transcript can also be attached to two events.
- **Headline timestamps.** Not stored, so leakage cannot be re-audited. The post-filter `assert_no_leakage` is dead code.
- **Split protocol.** Splits cut by row count, so same-date events can straddle splits. Nothing purges label windows that overlap the next split. Events are not deduplicated, and there is no coverage report.
- **Majority baseline.** Computed from the evaluation set's own labels, which is optimistic.
- **Metrics.**
  - A NaN validation AUC is silently scored as 0, which freezes early stopping at epoch 1.
  - AP, Brier score and log loss are missing.
  - Behaviour on empty input is undefined.
- **Provenance.** Model and tokenizer revisions are neither pinned nor recorded. Every run overwrites `artifacts/` and records no seed, mode, fingerprint or dependency information.
- **Documentation errors.**
  - The README calls the default head an "MLP". With `head_layers=0` it is dropout plus a single linear layer over `[t, n, |t−n|, t⊙n, presence flags]` (3,075 parameters for 768-d inputs), i.e. a logistic-regression head on fixed, expanded features.
  - The README says chunks are "≤512 tokens"; they are 384.
  - The docstring says chunks overlap; they do not.
- **Transcript truncation (method limitation).** Transcripts are capped at 8 chunks × 382 tokens, about 3k tokens. Real calls are usually longer, so the model mostly sees the prepared remarks.
- **Deprecated APIs.** `datetime.utcnow()` and `datetime.utcfromtimestamp()` are deprecated.

## Verification log
| # | Date | Command | Outcome |
|---|---|---|---|
| 1 | 2026-10-27 | `shasum -a 256` on the originals and the copies in `runs/legacy-2026-06-18-positive-control/` | All 5 files identical |
| 2 | 2026-10-27 | `git check-ignore -v project-spec.md .remember/… .venv data/cache/… artifacts/…` | All ignored |
| 3 | 2026-10-27 | `.venv/bin/python tests/smoke_test.py` (unmodified code) | **13/13 checks passed** in 12 s. Negative-control smoke test AUC = 0.073 (n=16; tiny sample, threshold not asserted). Positive-control test AUC = 1.000 |
| 4 | 2026-10-27 | scratch copy: `python main.py --synthetic --device cpu` | Bug 2 reproduced: loaded the cached negative-control events, then the positive-control embeddings. Neural test AUC 1.000; LR baseline AUC 1.000 |
| 5 | 2026-10-27 | scratch copy: `python main.py --device cpu` (real-data request) | Bug 1 reproduced: returned 400 cached synthetic events; test AUC 1.000 |
| 6 | 2026-10-27 | scratch copy: `python main.py --synthetic --no-cache --device cpu` | Bug 3 reproduced: fresh negative-control events, stale embeddings; test AUC 1.000 |
| 7 | 2026-10-27 | live yfinance probe (AAPL, MSFT, NVDA, JPM) | Timestamps in ET; AMC at 16:00, BMO at 06:00/07:00; the future row has NaN EPS. See [DATA_SOURCES.md](DATA_SOURCES.md) |
| 8 | 2026-10-27 | transformers 5.12.1 offline load of FinBERT; `torch.equal` on both snapshots | Loads commit `4556d13`; the `.bin` and safetensors weights are identical |
| 9 | 2026-10-27 | `.venv/bin/python -m pytest tests/unit/test_cache_regressions.py` (pre-fix code; committed as `3e5c9d1`) | **5/5 failed, as intended.** Failure messages: "real-data request returned synthetic events"; "positive-control request returned negative-control events"; "stale labels returned"; "stale embeddings returned"; "--no-cache still loaded the embedding cache" |
| 10 | 2026-10-27 | `.venv/bin/python -m pytest tests/unit/test_cache_regressions.py` (fixed code) | **11 passed.** The 5 scenarios from row 9, restated against the new API, plus: legacy `events.json` is never read, a tampered cache is an error, splits are not cached, an identical request hits, `--no-cache` writes nothing |
| 11 | 2026-10-27 | `.venv/bin/python -m pytest -m "not slow"` | **106 passed**, 1 deselected, in 3.6 s; no network |
| 12 | 2026-10-27 | `.venv/bin/python -m pytest -m slow` and `.venv/bin/python tests/smoke_test.py` | **1 passed**; script reports **13/13 checks**. Positive-control test AUC 1.000 (n=16); negative-control test AUC 0.109 (n=16, not asserted; see row 14) |
| 13 | 2026-10-27 | `python main.py --preset tiny --data-mode synthetic-positive-control --device cpu --no-cache --runs-dir <scratch>` | Run directory written with manifest, metrics, predictions and head. 77 events used, 3 purged. Neural val/test AUC 1.000/1.000; LR 1.000/1.000; majority 0.500 (positive control, as designed) |
| 14 | 2026-10-27 | Negative-control seed study (scratch only, not a saved experiment): tiny preset, `--no-cache`, seeds 100–139, n=80 and n=200 | Mean neural test AUC **0.504 ± 0.024 (SE)** at n=80 and **0.504 ± 0.015** at n=200; LR 0.496 / 0.494. The control is centred on chance. An earlier 10-seed look (seeds 0–9, n=80) gave 0.649 ± 0.048, a chance fluctuation that did not replicate on 40 fresh seeds |
| 15 | 2026-10-27 | Bug found by `test_controls_share_the_same_timeline` | The synthetic timing RNG consumed a mode-dependent number of draws, so the negative- and positive-control timelines differed. Fixed with a fixed-size draw. The test now passes |
| 16 | 2026-10-27 | `shasum -a 256 data/cache/* artifacts/*` after phase 2 | Legacy originals unchanged (same hashes as row 1) |
| 17 | 2026-10-27 | Review fixes: day-start news cutoff; `min_real_events` re-checked on a cache hit; tiny preset raised to 200 events | `.venv/bin/python -m pytest`: **109 passed** (108 unit + 1 slow) |
| 18 | 2026-10-27 | `uv pip compile requirements-lock.txt --python-platform x86_64-manylinux_2_28 --python-version 3.13` with the PyTorch CPU index | The lock resolves on Linux with `torch==2.12.1+cpu` and no `nvidia-*` packages. The CPU index lists `torch-2.12.1+cpu-cp313-…-manylinux_2_28_x86_64.whl` |
| 19 | 2026-10-27 | Clean clone of HEAD `9bde693` into scratch; `uv venv --python 3.13` (resolved to 3.13.15); `uv pip install -r requirements-lock.txt`; `HF_HOME=<empty scratch dir>` | Installed packages identical to the lock (`diff` empty); `project-spec.md` absent from the clone. `pytest -m "not slow"`: **108 passed**. `pytest -m slow`: **1 passed** (tiny BERT freshly downloaded). Tiny demos: negative-control test AUC neural 0.514 / LR 0.432 (n=40); positive-control 1.000 / 1.000 |
| 20 | 2026-10-27 | Reproducibility: both tiny demos run twice in the dev env with `--no-cache`, compared with the clean-clone runs | `metrics.json` metrics **identical** across repeats and across the two environments (Python 3.13.17 vs 3.13.15); dataset fingerprints identical |
| 21 | 2026-10-27 | The CI workflow's exact install and test commands (pip, `torch==2.12.1` from the CPU index, then the lock) in a fresh pip venv on macOS | torch 2.12.1; **108 passed**; **1 passed** (slow); both demos wrote run directories. The workflow on GitHub-hosted Ubuntu has **not run yet (pending first push)** |
| 22 | 2026-10-27 | Clean clone: `HF_HUB_OFFLINE=1 HF_HOME=<new empty dir> .venv/bin/python -m pytest -m "not slow" --durations=8` | **108 passed in 8.1 s, offline.** The HF directory stayed empty, so the unit tests need no network. The slowest test took 2.2 s; the earlier 41 s / 29 s fresh-venv runs were cold-start imports |
| 23 | 2026-10-27 | Documented alternative install: fresh venv, `uv pip install -r requirements-dev.txt` (compatible ranges) | Resolved to newer releases (torch 2.14.1, transformers 5.15.1, huggingface-hub 1.33.0, yfinance 1.7.0, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1). `pytest`: **109 passed** (including slow) |
| 24 | 2026-10-27 | Added: the embedding cache key includes the encoding device; a test that real-mode `--no-cache` / `--refresh-cache` builds providers without the raw-response cache | `.venv/bin/python -m pytest`: **112 passed** (111 unit + 1 slow) |
| 25 | 2026-10-27 | `shasum -a 256` on the legacy originals and the copies in `runs/legacy-…`, end of phase 3 | All identical to row 1 |
| 26 | 2026-10-27 | Key check: `.env` permissions `-rw-------`, gitignored; `source .env` | Both variables set (lengths only checked, values never printed) |
| 27 | 2026-10-27 | Live provider probes via `earnings_classifier.providers` (key-redacting client, no raw cache) | **Finnhub:** news present back to about 11 months, none at about 13 months or 2 years. **FMP:** `profile` OK (key valid); transcript-dates and transcript endpoints → HTTP 402 Payment Required |
| 28 | 2026-10-27 | Added `LICENSE` (MIT); `uv build --wheel` | Wheel metadata `License-Expression: MIT`, `License-File: LICENSE` |
| 29 | 2026-10-27 | Document-stream refactor plus EDGAR provider; `.venv/bin/python -m pytest` | **129 passed** (128 unit + 1 slow). New tests: EDGAR filtering (8-K only, Item 2.02 exact, amendments excluded), UTC→ET acceptance dates across DST and the evening date roll, older submission pages fetched only when needed, EX-99.1 preference, text extraction, the User-Agent sent but never cached, provider config errors, the earlier news cutoff, short/empty documents, one document per event, legacy `transcript_*` config names |
| 30 | 2026-10-27 | Live EDGAR probe (contact User-Agent from `.env`, never printed) for AAPL, MSFT, JPM against their Yahoo announcements | **49/49 matched on the same ET date for each ticker.** Acceptance hours agree with the Yahoo sessions (AAPL/MSFT 16:xx, JPM 06:xx–07:xx). Texts ~10k–49k characters. 78 requests at ≤8.3/s. yfinance returned 49 rows for `limit=40`; the collector now keeps the 40 most recent |
| 31 | 2026-10-27 | Coverage dry run: `main.py --data-mode real --collect-only` (30 tickers × 40 announcements; SEC EDGAR + Finnhub; 10:35–11:02, about 27 min, mostly Finnhub's ~1 request/s) | 1,200 announcements → **1,138 usable**, 0 provider errors. Modality: both 109, document-only 1,027, headlines-only 2. **Headlines occur only in test** (train/val: 0). Missing documents: 51 `no_document_in_window`, 12 `empty_document`. Medians: 29k characters per release |
| 32 | 2026-10-27 | Diagnosis of the missing documents (raw cache plus a few live EDGAR calls; Yahoo timed out/reset twice) | XOM: one Item 2.02 8-K since 2016. DIS: current CIK's filings start 2019. JNJ: release typed `EX-99.15`. KO/NVDA/COST: 4 events without an 8-K within [-1, +3] days. Fixes: first-`EX-99.*` fallback, Yahoo retries (3 tries, 3 s/6 s back-off, also on empty frames) |
| 33 | 2026-10-27 | Audit of every chosen exhibit, then the description-based rule (NVDA `EX-95.1`); `pytest -m "not slow"`; both variants re-collected from the raw cache | **131 passed.** Both: **1,150 usable** (fingerprint `ce118a67e773bc89…`, `events-real-51bdfd00865f9458.json`). Document-only (`--headlines none`): **1,148 usable** (fingerprint `2caa9d48aedd07ab…`, `events-real-88412e2753d3aca7.json`); splits train 740 (2016-10-19..2023-04-24, up 0.470), val 169 (2023-04-27..2024-10-24, up 0.527), test 229 (2024-10-31..2026-10-23, up 0.454); purged 6 + 4. 0 errors in both |
| 34 | 2026-10-27 | Froze both datasets: copied to `data/frozen/` (read-only, git-ignored); `shasum -a 256` | `events-real-document-only.json` sha256 `7c916e19c143e7aaeb9def18b9113308dfcc4239b1a5f905ab2dda6dab99b1a3` (fingerprint `2caa9d48aedd07abb76bead601c3ddec026f6493edbb9fcf429defbf69e98259`, 1,148 events); `events-real-with-headlines.json` sha256 `14a8db89e726f46f43cbc37c378dca3ef22f8e96b8d4ebd68610695d1ee37386` (fingerprint `ce118a67e773bc89b867d1965dac2ae9242f008eb776c3dc9fcdc5b96f5adcfa`, 1,150 events). Identical to the cache files |
| 35 | 2026-10-27 | `main.py --collect-only --frozen-events data/frozen/events-real-document-only.json --expect-fingerprint 2caa9d48… --val-start-date 2023-04-27 --test-start-date 2024-10-31`; and with `--expect-fingerprint deadbeef` | Explicit dates reproduce **740 / 169 / 229, purged 6 + 4**. The wrong fingerprint aborts with `DataValidationError` before any processing |
| 36 | 2026-10-27 | Phase 4 tooling (bootstrap, ticker prior, zero-shot, selection guard, frozen loading, runner helpers); `.venv/bin/python -m pytest` | **147 passed** (146 unit + 1 slow) |
| 37 | 2026-10-27 | FinBERT controls from a snapshot worktree at `d286ca0`: `main.py --data-mode {negative,positive} --seed 0..4 --runs-dir runs/controls-finbert`; `experiments/summarize_controls.py` | 10/10 runs exit 0. Negative control test AUC **0.465 ± 0.075** (0.392–0.578); positive **1.000** on all 5 seeds |
| 38 | 2026-10-27 | `experiments/phase4.py select` (validation only; embeddings on MPS, 1 chunk ~1.5 min, 8 chunks ~5 min; heads on CPU) | Selected LR 1 chunk C=0.01 (val 0.539), head 1 chunk linear (val 0.602 ± 0.008). The selection file has no test entries. Committed as `e5667c9` |
| 39 | 2026-10-27 | `experiments/phase4.py test` (passed the `require_committed` guard for `e5667c9`) | Test AUC: majority 0.500, ticker prior 0.487, LR 0.461, fusion head **0.506 [0.439, 0.573]**, zero-shot 0.469. No paired difference excludes 0. Committed as `939ce85` |
| 40 | 2026-10-27 | `experiments/phase4.py headlines` (exploratory, training-free) | 111 events: headline sentiment AUC 0.429 [0.323, 0.556]; on the 109 with both, release 0.413, difference +0.022 [−0.127, +0.182] |
| 41 | 2026-10-27 | `brew install tectonic` | tectonic 0.17.0 installed |
| 42 | 2026-10-27 | Bibliography verification: Crossref (16 DOIs), arXiv API (3), jmlr.org, nber.org, sec.gov (with the contact User-Agent), openlibrary.org, Hugging Face model cards at the pinned revisions | All entries confirmed. Three end pages and one subtitle that Crossref did not return were removed rather than guessed. FinBERT model card: further-trained BERT, fine-tuned on Financial PhraseBank; no license field. BERT-tiny: Apache-2.0 |
| 43 | 2026-10-27 | `experiments/dataset_summary.py`, `experiments/paper_assets.py`, `tectonic -X compile paper/main.tex`; pages rendered with PDFKit and inspected | 11 pages, no TeX warnings. Fixed during inspection: a legend overlap, timeline label collisions, a missing `\mathbb` glyph, caption spacing, and one hand-typed range replaced by macros. Re-running the generators leaves every `.tex` file and `tables.md` unchanged |
| 44 | 2026-10-27 | `git grep` over tracked files for `/Users/`, the username, and the email; byte search of tracked files for the `.env` values | Only the 10 FinBERT control manifests contained the local path (`argv`, `runs_dir`). They were sanitized to repo-relative paths with a `publication_note`, and `build_manifest` now relativizes paths (test added). No email or key value in any tracked file, including the PDFs |

