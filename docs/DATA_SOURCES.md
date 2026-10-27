# Data sources: verified behaviour

This file records what was checked against live services or official documentation, and when. Anything marked **unverified** needs a live check with credentials before it is relied on.

## Yahoo Finance via `yfinance` 1.4.1 (prices, earnings dates; no key)
Probed live on 2026-10-27 for AAPL, MSFT, NVDA and JPM using `Ticker.get_earnings_dates(limit=16)` and `Ticker.history(auto_adjust=True)`.

- **Earnings-date index.** `DatetimeIndex` in tz `America/New_York`. Columns: `EPS Estimate`, `Reported EPS`, `Surprise(%)`.
- **Time of day.**
  - After-close reporters (AAPL, MSFT, NVDA) are stamped `16:00` ET.
  - Pre-open reporters (JPM) are stamped `06:00`/`07:00` ET.
  - Times are provider approximations, not the exact release minute.
- **Upcoming events.** Each list includes the next scheduled event, with `Reported EPS = NaN` and placeholder times such as `15:00`. Only rows with a reported EPS **and** a timestamp in the past count as realized events.
- **Daily bars.** Indexed at midnight `America/New_York`, with `Open` and `Close` adjusted (`auto_adjust=True`).
- **Implication for the old code.** `_to_naive_utc(ts).date()` converts ET to UTC before taking the date. For any timestamp at or after 19:00/20:00 ET (depending on DST), that rolls the date forward one day. Session logic must run in ET.
- **Redistribution.** Yahoo data is subject to Yahoo's terms. Raw responses are not committed; test fixtures are hand-made with the fake ticker `ACME`.

  - `get_earnings_dates(limit=40)` returned **49** realized rows (2014-07..2026-07) for AAPL, MSFT and JPM, so the provider can return more rows than requested. The collector keeps the `earnings_per_ticker` most recent ones.

## SEC EDGAR (earnings press releases; default document stream; `SEC_EDGAR_USER_AGENT`)
Free, official, and exactly timestamped. Probed live on 2026-10-27 with a contact User-Agent.

- **Access policy.** `https://www.sec.gov/os/accessing-edgar-data`: every request must declare a User-Agent with a contact (name + email), and traffic is capped at 10 requests/second. The client sends `SEC_EDGAR_USER_AGENT` from the environment as that header and spaces requests 0.12 s apart. The value is never written to caches, manifests or the repository.
- **Endpoints used.**
  - `https://www.sec.gov/files/company_tickers.json`: ticker → CIK (`ticker`, `cik_str`). Class shares use a dash (`BRK-B`).
  - `https://data.sec.gov/submissions/CIK##########.json`: filing history. `filings.recent` holds parallel column arrays (`accessionNumber`, `acceptanceDateTime`, `form`, `items`, `primaryDocument`, ...). Older filings live in the pages listed under `filings.files` (`name`, `filingFrom`, `filingTo`), which have the same columns. Pages that end before the first needed date are not requested.
  - Filing index `https://www.sec.gov/Archives/edgar/data/<cik>/<accession without dashes>/<accession>-index.htm`: a table of Seq / Description / Document / Type / Size.
- **Which filings count.** Form `8-K` (amendments `8-K/A` excluded) whose `items` list contains `2.02` ("Results of Operations and Financial Condition"). The text is the exhibit typed `EX-99.1`, else `EX-99.01`, else `EX-99`, else the first `EX-99.*` in filing order (see JNJ below). A filing without any `EX-99` exhibit has no text.
- **Release time.** `acceptanceDateTime` is true UTC with a `Z` suffix (e.g. AAPL's 16:30 ET release reads `20:30Z` in summer and `21:30Z` in winter). The document date is the **ET** date of acceptance: an 8-K accepted at `03:10Z` belongs to the previous ET evening.
- **Text extraction.** Visible text only: EDGAR's SGML header (`<TYPE>`, `<SEQUENCE>`, `<FILENAME>`, `<DESCRIPTION>`), `<title>`, `<script>` and `<style>` are dropped, and text nodes are joined with spaces so table cells never fuse into one word. Exhibits shorter than `min_document_chars` (200) count as missing.
- **Verified live (2026-10-27).** The 40 most recent Yahoo announcements of AAPL, MSFT and JPM were each matched to an 8-K Item 2.02 on the **same ET date** (49/49 before truncation, lag 0 days for every one).
  - Acceptance hours (ET): AAPL 89 of 94 at 16:xx; MSFT 69 of 72 at 16:xx; JPM 44 of 49 at 06:xx and 4 at 07:xx. These agree with the Yahoo session stamps.
  - History: AAPL's earliest Item 2.02 is 2004-10-13. The SEC's final rule "Additional Form 8-K Disclosure Requirements and Acceleration of Filing Date" (69 FR 15593, effective August 23, 2004) was fetched from sec.gov on 2026-10-27; the paper cites it only for the item numbering. Whether earnings releases were furnished under another item before then was not checked.
  - Extracted lengths: AAPL ~10-11k characters, MSFT ~12-23k, JPM ~23-49k. Across the 30-ticker dry run: min 5,794, **median 29,016**, max 99,818 characters (n = 1,136). With 8 chunks of 382 tokens, only roughly the first 3k tokens (~12k characters) are encoded: the headline results, the narrative and the outlook, which come first, but usually not the financial tables later in the release.
- **Coverage gaps found by the 30-ticker dry run (2026-10-27).**
  - **XOM** filed only one Item 2.02 8-K since 2016 (its other 8-Ks are Items 7.01/8.01), so ExxonMobil's earnings releases are not on EDGAR in this form. XOM keeps only headline-bearing recent events.
  - **DIS** filings under the CIK that `company_tickers.json` maps today start in 2019. The earlier releases are presumably under the pre-2019 registrant (the Fox-acquisition holding-company change; not verified), so 2016–2019 DIS events have no document. Former CIKs are not followed.
  - **JNJ** typed its 2016–2019 release exhibit `EX-99.15` (and the tables `EX-99.2O`), described only as "EXHIBIT 99.15". The 8-K cover lists 99.15 as "Press Release dated …".
  - **NVDA** typed its 2019-11-14 release `EX-95.1` (described "Q3FY20 PRESS RELEASE"), next to an `EX-99.2` CFO commentary. A first-`EX-99.*` rule picked the commentary; this was caught in the audit below.
  - **Selection rule therefore:** `EX-99.1` / `EX-99.01` / `EX-99`; else an `EX-*` exhibit whose description says "press release", "earnings release" or "news release"; else the first `EX-99.*` in filing order. The coverage report counts the exhibit type each fetch used. In the final dry run: `EX-99.1` 1,131, `EX-99.15` 11, `EX-99` 4, `EX-99.01` 2, `EX-95.1` 1.
  - **Audit of all 1,148 chosen exhibits** (from the raw cache): 906 are described by exhibit number, 189 as a press/earnings/news release, 13 as "INVESTOR RELEASE", 9 as MSFT-style "FY… 8-K EXHIBIT 99.1". The 51 whose first 400 characters lack words like "reports", "results" or "quarter" were read by ticker. They are NFLX shareholder letters (24), BAC "Financial Highlights" releases (11), ORCL (12) and ADBE (1) releases opening with media contacts, and WMT (3) releases opening with comp-sales figures. All are earnings releases. Formats differ across issuers (TSLA and NFLX publish shareholder letters/decks as the release).
  - **KO, NVDA, COST**: 4 events in total have no Item 2.02 8-K within [-1, +3] days of the Yahoo date.
- **Yahoo flakiness.** During the diagnosis, yfinance calls intermittently timed out or reset (`curl: (28)`, `(35)`). Yahoo calls are now retried up to 3 times (3 s, then 6 s back-off), including on empty frames, so a transient failure cannot silently drop a ticker. Yahoo responses are not raw-cached, so the frozen events cache, not re-collection, is what makes a dataset reproducible.
- **What the text is.** A company-authored press release, published at the announcement. It is not the earnings-call transcript, which adds management Q&A and comes later. Results using this stream are about press-release language, not call language.
- **Redistribution.** Raw responses stay in the git-ignored `data/raw/`; tests use hand-made fixtures for the fake issuer ACME.

## Finnhub `company-news` (headlines; `FINNHUB_API_KEY`)
Checked against the official OpenAPI spec (`https://finnhub.io/static/swagger.json`, fetched 2026-10-27).

- **Request.** `GET /api/v1/company-news?symbol=&from=YYYY-MM-DD&to=YYYY-MM-DD&token=` (North American companies only).
- **Response fields.** `category`, `datetime` (published time, UNIX seconds), `headline`, `id`, `image`, `related`, `source`, `summary`, `url`.
- **Free tier:** "1 year of historical news and new updates". Events older than about a year therefore cannot get headlines on the free tier.
- **Finnhub transcripts.** `/stock/transcripts` and `/stock/transcripts/list` are marked "Premium required", so they are not usable without a paid plan.
- **Rate limit.** Secondary sources say about 60 calls/min on the free tier (unverified against official docs).
- **Verified live with a free-tier key (2026-10-27, AAPL, 3-day windows):**
  - 201 headlines for 2026-10-20..23.
  - 108 headlines for 2025-11-26..29 (about 11 months back).
  - **0** for 2025-09-27..30 (about 13 months back) and 0 for 2024-10-27..30.

  History therefore ends about 12 months before today. Busy names produce far more headlines per window than `max_headlines` (16), so only the most recent 16 before the cutoff are encoded.

## Financial Modeling Prep (transcripts; `FMP_API_KEY`)
The official docs (site.financialmodelingprep.com via the Context7 index, 2026-10-27) list the "stable" endpoints:

- `GET https://financialmodelingprep.com/stable/earning-call-transcript-dates?symbol=AAPL`: transcript availability by fiscal year, quarter and date.
- `GET https://financialmodelingprep.com/stable/earning-call-transcript?symbol=AAPL&year=2020&quarter=3`: the full transcript.
- The legacy `api/v3/earning_call_transcript/{ticker}?quarter=&year=` used by the pre-audit code is not the documented current endpoint.
- **Verified live with a free-tier key (2026-10-27):**
  - The key is valid: `GET /stable/profile?symbol=AAPL` returns data.
  - `earning-call-transcript-dates` and `earning-call-transcript` both return **HTTP 402 (Payment Required)**. Transcripts are not available on the free plan.
  - Third-party pricing summaries put them on the top paid tier; this was not verified on FMP's own pricing page.
- **Still unverified (would need a plan that includes transcripts):**
  - The exact response field names. Expected: `fiscalYear`, `quarter`, `date` for the dates list; `symbol`, `period`, `year`, `date`, `content` for transcripts.
  - Which plan tier includes transcripts.
  - Whether `date` carries a time of day.

  The parser fails loudly on any schema mismatch instead of guessing.

## Pretrained encoders (Hugging Face Hub)
- **`ProsusAI/finbert`.** Pinned to commit `4556d13015211d73dccd3fdd39d39232506f3e43`, which is `refs/main` as of 2026-10-27; transformers 5.12.1 reports this `_commit_hash`.
  - The local cache also holds snapshot `7db323f…` (a safetensors conversion).
  - Its 201 tensors are bit-identical to `pytorch_model.bin` at `4556d13`, checked with `torch.equal` on 2026-10-27.
- **`google/bert_uncased_L-2_H-128_A-2`.** Tiny 2-layer BERT for tests and demos, pinned to `30b0a37ccaaa32f332884b96992754e246e48c5f`.
