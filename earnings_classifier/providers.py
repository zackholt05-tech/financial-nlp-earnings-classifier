"""Data-provider access.

  * Yahoo Finance via yfinance  - daily prices and earnings timestamps (no key)
  * SEC EDGAR                    - earnings press releases: Exhibit 99.1 of 8-K
                                   filings that report Item 2.02 (free; needs a
                                   contact User-Agent, SEC_EDGAR_USER_AGENT)
  * Financial Modeling Prep      - earnings-call transcripts (FMP_API_KEY; paid plan)
  * Finnhub                      - company-news headlines (FINNHUB_API_KEY)

Both document providers expose the same interface (`list_documents`,
`fetch_text`) and return `DocumentRef`s, so the collector treats "the
company's earnings document" uniformly and records which kind it was.

Rules enforced here:
  * Credentials come only from environment variables and are redacted from every
    error message; the raw-response cache is keyed by the URL *without* the key
    and never stores request headers.
  * Failures are loud. Auth/plan errors, rate limits, network errors and
    unexpected response schemas raise `ProviderError`; only a well-formed
    "no data" answer (empty list / 404) is treated as missing data.
Endpoints and fields are documented in docs/DATA_SOURCES.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from html import unescape
from html.parser import HTMLParser
from typing import Callable, Dict, List, Optional, Tuple

from .config import DOCUMENT_PROVIDERS, HEADLINE_PROVIDERS
from .timing import ET, to_et, utc_from_unix


class ProviderError(RuntimeError):
    """A provider request failed or returned something we cannot trust."""


class ProviderAuthError(ProviderError):
    """Invalid key, or the endpoint is not included in the account's plan."""


class ProviderRateLimitError(ProviderError):
    pass


class ProviderConfigError(ProviderError):
    """A requested provider is missing its credential / contact setting."""


def redact(text: str, secrets) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


class JsonHttp:
    """Minimal HTTPS client: optional secret query parameter, fixed request
    headers, credential redaction, throttling and an optional raw-response cache."""

    def __init__(self, provider: str, secret_param: Optional[str], secret: str,
                 raw_cache_dir: Optional[str] = None, timeout: float = 30.0,
                 min_interval: float = 0.0, opener: Callable = urllib.request.urlopen,
                 headers: Optional[Dict[str, str]] = None):
        self.provider = provider
        self.secret_param = secret_param
        self.secret = secret
        self.headers = dict(headers or {})
        self.raw_cache_dir = raw_cache_dir
        self.timeout = timeout
        self.min_interval = min_interval
        self.opener = opener
        self._last = 0.0
        self.requests = 0
        self.cache_hits = 0

    def _secrets(self) -> List[str]:
        return [self.secret, *self.headers.values()]

    def _cache_path(self, public_url: str) -> Optional[str]:
        if not self.raw_cache_dir:
            return None
        h = hashlib.sha256(public_url.encode()).hexdigest()
        return os.path.join(self.raw_cache_dir, self.provider, f"{h}.json")

    def _fetch(self, base_url: str, params: Dict) -> Optional[str]:
        """Raw response text, or None for 404."""
        query = urllib.parse.urlencode(sorted(params.items()))
        public_url = f"{base_url}?{query}" if query else base_url
        path = self._cache_path(public_url)
        if path and os.path.exists(path):
            with open(path) as f:
                rec = json.load(f)
            if rec.get("url") != public_url:
                raise ProviderError(f"raw cache collision at {path}")
            self.cache_hits += 1
            return rec["text"]

        full_params = dict(params)
        if self.secret_param:
            full_params[self.secret_param] = self.secret
        full_query = urllib.parse.urlencode(sorted(full_params.items()))
        full_url = f"{base_url}?{full_query}" if full_query else base_url
        wait = self.min_interval - (_time.monotonic() - self._last)
        if wait > 0:
            _time.sleep(wait)
        self._last = _time.monotonic()
        self.requests += 1
        request = urllib.request.Request(full_url, headers=self.headers)
        try:
            with self.opener(request, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            msg = redact(f"{self.provider} HTTP {e.code} for {public_url}", self._secrets())
            if e.code == 404:
                return None
            if e.code == 402:
                raise ProviderAuthError(msg + " (payment required: endpoint not included in this "
                                        "plan; choose another provider for this stream)") from None
            if e.code in (401, 403):
                raise ProviderAuthError(msg + " (invalid key/contact header, or endpoint not "
                                        "in plan)") from None
            if e.code == 429:
                raise ProviderRateLimitError(msg + " (rate limited)") from None
            raise ProviderError(msg) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ProviderError(redact(f"{self.provider} request failed for {public_url}: {e}",
                                       self._secrets())) from None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump({"url": public_url, "provider": self.provider,
                           "fetched_at": datetime.now(timezone.utc).isoformat(),
                           "text": raw}, f)
        return raw

    def get(self, base_url: str, params: Dict) -> Optional[object]:
        raw = self._fetch(base_url, params)
        if raw is None:
            return None
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            raise ProviderError(f"{self.provider} returned non-JSON for {base_url}") from None
        if isinstance(body, dict) and ("Error Message" in body or "error" in body):
            detail = redact(str(body.get("Error Message") or body.get("error")), self._secrets())
            raise ProviderAuthError(f"{self.provider} refused {base_url}: {detail}")
        return body

    def get_text(self, url: str) -> Optional[str]:
        return self._fetch(url, {})


# ---------------------------------------------------------------------------
# Earnings documents (shared reference type)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DocumentRef:
    key: str                          # unique id: SEC accession number, or "FY2024Q2"
    doc_date: date                    # ET calendar date the document became public
    doc_time: Optional[datetime] = None   # aware UTC, when the provider gives an exact time
    fiscal_year: Optional[int] = None     # provider's fiscal period, when it states one
    fiscal_quarter: Optional[int] = None
    url: str = ""


def _parse_date(value, provider: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        raise ProviderError(f"{provider}: unparseable date {value!r}") from None


# ---------------------------------------------------------------------------
# Financial Modeling Prep: earnings-call transcripts (paid plan)
# ---------------------------------------------------------------------------
class FMPTranscripts:
    BASE = "https://financialmodelingprep.com/stable"
    source = "fmp"
    document_type = "earnings_call_transcript"

    def __init__(self, http: JsonHttp):
        self.http = http

    @staticmethod
    def ref(fiscal_year: int, quarter: int, call_date: date) -> DocumentRef:
        return DocumentRef(f"FY{fiscal_year}Q{quarter}", call_date, None, fiscal_year, quarter)

    def list_documents(self, symbol: str, since: Optional[date] = None) -> List[DocumentRef]:
        body = self.http.get(f"{self.BASE}/earning-call-transcript-dates", {"symbol": symbol})
        if body is None:
            return []
        if not isinstance(body, list):
            raise ProviderError(f"fmp: expected a list of transcript dates for {symbol}")
        out = []
        for row in body:
            try:
                out.append(self.ref(int(row["fiscalYear"]), int(row["quarter"]),
                                    _parse_date(row["date"], "fmp")))
            except (KeyError, TypeError, ValueError) as e:
                raise ProviderError(f"fmp: unexpected transcript-dates row for {symbol}: "
                                    f"missing/invalid {e}") from None
        return out

    def fetch_text(self, symbol: str, ref: DocumentRef) -> str:
        body = self.http.get(f"{self.BASE}/earning-call-transcript",
                             {"symbol": symbol, "year": ref.fiscal_year,
                              "quarter": ref.fiscal_quarter})
        if not body:
            return ""
        if not isinstance(body, list) or not isinstance(body[0], dict) or "content" not in body[0]:
            raise ProviderError(f"fmp: unexpected transcript payload for {symbol} {ref.key}")
        row = body[0]
        if "date" in row and _parse_date(row["date"], "fmp") != ref.doc_date:
            raise ProviderError(f"fmp: transcript date {row['date']} does not match the listed "
                                f"call date {ref.doc_date} for {symbol} {ref.key}")
        return (row["content"] or "").strip()


# ---------------------------------------------------------------------------
# SEC EDGAR: earnings press releases (8-K, Item 2.02, Exhibit 99.1)
# ---------------------------------------------------------------------------
_SGML_HEADER = re.compile(r"<(TYPE|SEQUENCE|FILENAME|DESCRIPTION)>[^\n<]*", re.I)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_HREF = re.compile(r'href="([^"]+)"', re.I)
_TAG = re.compile(r"<[^>]+>")
_RELEASE = re.compile(r"\b(PRESS|EARNINGS|NEWS)\s+RELEASE\b")


class _VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "title"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "title") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    """Visible text of an EDGAR document (HTML or SGML-wrapped plain text),
    whitespace-normalized. Text nodes are joined with spaces so words in
    adjacent cells/tags never fuse; the SGML <TYPE>/<FILENAME>/... header that
    EDGAR prepends to exhibits is dropped."""
    parser = _VisibleText()
    parser.feed(_SGML_HEADER.sub(" ", raw))
    parser.close()
    return re.sub(r"\s+", " ", " ".join(parser.parts).replace("\xa0", " ")).strip()


def exhibit_href(index_html: str,
                 preferred=("EX-99.1", "EX-99.01", "EX-99")) -> Optional[Tuple[str, str]]:
    """(exhibit type, link) of the press-release exhibit in an EDGAR filing index.

    Order: the standard types in `preferred`; else an exhibit whose description
    names a press/earnings release (NVIDIA's 2019-11-14 release is typed
    "EX-95.1"); else the first EX-99.* exhibit in filing order (Johnson &
    Johnson's 2016-2019 releases are typed "EX-99.15", described only as
    "EXHIBIT 99.15"). None if no exhibit qualifies.
    """
    found: Dict[str, str] = {}
    described: List[Tuple[str, str]] = []
    other: List[Tuple[str, str]] = []
    for row in _ROW.findall(index_html):
        cells = _CELL.findall(row)
        if len(cells) < 4:
            continue
        clean = [unescape(_TAG.sub("", c)).replace("\xa0", " ").strip().upper() for c in cells]
        doc_type, description = clean[3], clean[1]
        href = _HREF.search(cells[2])
        if not href or not doc_type.startswith("EX-"):
            continue
        link = href.group(1).replace("/ix?doc=", "", 1)
        if doc_type in preferred:
            found.setdefault(doc_type, link)
        elif _RELEASE.search(description):
            described.append((doc_type, link))
        elif doc_type.startswith("EX-99"):
            other.append((doc_type, link))
    for t in preferred:
        if t in found:
            return t, found[t]
    for pool in (described, other):
        if pool:
            return pool[0]
    return None


class SecEdgarPressReleases:
    source = "sec-edgar"
    document_type = "earnings_press_release"
    TICKERS = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS = "https://data.sec.gov/submissions"
    SEC = "https://www.sec.gov"

    def __init__(self, http: JsonHttp):
        self.http = http
        self._ciks: Optional[Dict[str, int]] = None
        self.exhibit_types: Counter = Counter()   # which exhibit type each fetch used

    def cik(self, symbol: str) -> int:
        if self._ciks is None:
            body = self.http.get(self.TICKERS, {})
            if not isinstance(body, dict):
                raise ProviderError("sec: unexpected company_tickers.json payload")
            self._ciks = {str(v["ticker"]).upper(): int(v["cik_str"]) for v in body.values()}
        key = symbol.upper().replace(".", "-")
        if key not in self._ciks:
            raise ProviderError(f"sec: no CIK for ticker {symbol}")
        return self._ciks[key]

    def _filing_rows(self, cik: int, since: Optional[date]) -> List[Dict]:
        sub = self.http.get(f"{self.SUBMISSIONS}/CIK{cik:010d}.json", {})
        if not isinstance(sub, dict) or "filings" not in sub:
            raise ProviderError(f"sec: unexpected submissions payload for CIK {cik}")
        blocks = [sub["filings"]["recent"]]
        for extra in sub["filings"].get("files", []):
            if since is None or _parse_date(extra["filingTo"], "sec") >= since:
                blocks.append(self.http.get(f"{self.SUBMISSIONS}/{extra['name']}", {}))
        rows = []
        for b in blocks:
            try:
                cols = [b["accessionNumber"], b["acceptanceDateTime"], b["form"], b["items"]]
            except (KeyError, TypeError):
                raise ProviderError(f"sec: submissions block without the expected columns "
                                    f"for CIK {cik}") from None
            rows += [dict(zip(("accession", "accepted", "form", "items"), r)) for r in zip(*cols)]
        return rows

    def list_documents(self, symbol: str, since: Optional[date] = None) -> List[DocumentRef]:
        cik = self.cik(symbol)
        out, seen = [], set()
        for r in self._filing_rows(cik, since):
            if r["form"] != "8-K" or "2.02" not in str(r["items"]).split(","):
                continue
            if r["accession"] in seen:
                continue
            seen.add(r["accession"])
            try:
                accepted = datetime.fromisoformat(str(r["accepted"]).replace("Z", "+00:00"))
            except ValueError:
                raise ProviderError(f"sec: bad acceptanceDateTime {r['accepted']!r}") from None
            if accepted.tzinfo is None:
                raise ProviderError(f"sec: acceptanceDateTime without zone {r['accepted']!r}")
            acc = r["accession"]
            url = (f"{self.SEC}/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"
                   f"{acc}-index.htm")
            out.append(DocumentRef(acc, accepted.astimezone(ET).date(),
                                   accepted.astimezone(timezone.utc), None, None, url))
        return sorted(out, key=lambda d: d.doc_time)

    def fetch_text(self, symbol: str, ref: DocumentRef) -> str:
        index = self.http.get_text(ref.url)
        if index is None:
            raise ProviderError(f"sec: filing index not found for {symbol} {ref.key}")
        choice = exhibit_href(index)
        self.exhibit_types[choice[0] if choice else "none"] += 1
        if choice is None:
            return ""  # 8-K Item 2.02 without a press-release exhibit
        href = choice[1]
        raw = self.http.get_text(href if href.startswith("http") else self.SEC + href)
        if raw is None:
            raise ProviderError(f"sec: exhibit missing for {symbol} {ref.key}")
        return html_to_text(raw)


# ---------------------------------------------------------------------------
# Finnhub: company news headlines
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Headline:
    text: str
    published: datetime   # aware, UTC
    source: str


class FinnhubNews:
    BASE = "https://finnhub.io/api/v1"
    source = "finnhub"

    def __init__(self, http: JsonHttp):
        self.http = http

    def company_news(self, symbol: str, start: date, end: date) -> List[Headline]:
        body = self.http.get(f"{self.BASE}/company-news",
                             {"symbol": symbol, "from": start.isoformat(), "to": end.isoformat()})
        if body is None:
            return []
        if not isinstance(body, list):
            raise ProviderError(f"finnhub: expected a list of news items for {symbol}")
        out = []
        for row in body:
            try:
                ts, text = row["datetime"], (row["headline"] or "").strip()
            except (KeyError, TypeError):
                raise ProviderError(f"finnhub: news item without datetime/headline for {symbol}") \
                    from None
            if text:
                out.append(Headline(text, utc_from_unix(ts), str(row.get("source", ""))))
        return out


# ---------------------------------------------------------------------------
# Yahoo Finance via yfinance: earnings dates + daily prices (no key)
# ---------------------------------------------------------------------------
def _with_retries(fetch: Callable, what: str, attempts: int = 3, wait: float = 3.0,
                  sleep: Callable = _time.sleep):
    """Call `fetch` until it returns a non-empty result. Yahoo intermittently
    times out or returns an empty frame for valid tickers; without retries a
    transient failure would silently drop a ticker from the dataset."""
    last_exc, result = None, None
    for k in range(attempts):
        try:
            result = fetch()
            last_exc = None
            if result is not None and not getattr(result, "empty", False):
                return result
        except Exception as e:  # noqa: BLE001 - yfinance raises many ad-hoc errors
            last_exc = e
        if k + 1 < attempts:
            sleep(wait * (k + 1))
    if last_exc is not None:
        raise ProviderError(f"yahoo: {what} failed after {attempts} attempts: {last_exc}") \
            from None
    return result


class YahooMarketData:
    source = "yahoo"

    def __init__(self, sleep: Callable = _time.sleep):
        self.sleep = sleep

    def earnings_times(self, ticker: str, limit: int) -> List[datetime]:
        """The `limit` most recent past, realized announcement timestamps (ET-aware),
        oldest first. (yfinance may return more rows than requested.)"""
        import pandas as pd
        import yfinance as yf

        df = _with_retries(lambda: yf.Ticker(ticker).get_earnings_dates(limit=limit),
                           f"earnings dates for {ticker}", sleep=self.sleep)
        if df is None or df.empty:
            return []
        if "Reported EPS" not in df.columns:
            raise ProviderError(f"yahoo: no 'Reported EPS' column for {ticker}")
        now = datetime.now(timezone.utc)
        out = sorted(to_et(ix) for ix, eps in zip(df.index, df["Reported EPS"])
                     if pd.notna(eps) and to_et(ix) < now)
        return out[-limit:]

    def daily_bars(self, ticker: str, start: date, end: date):
        import yfinance as yf

        df = _with_retries(lambda: yf.Ticker(ticker).history(start=start.isoformat(),
                                                            end=end.isoformat(),
                                                            auto_adjust=True),
                           f"price history for {ticker}", sleep=self.sleep)
        if df is None or df.empty:
            return None
        return df[["Open", "Close"]]


@dataclass
class Providers:
    market: YahooMarketData
    documents: Optional[object]       # SecEdgarPressReleases | FMPTranscripts | None
    news: Optional[FinnhubNews]

    @classmethod
    def from_env(cls, raw_cache_dir: Optional[str], documents: str = "sec-edgar",
                 headlines: str = "finnhub") -> "Providers":
        if documents not in DOCUMENT_PROVIDERS:
            raise ProviderConfigError(f"documents must be one of {DOCUMENT_PROVIDERS}")
        if headlines not in HEADLINE_PROVIDERS:
            raise ProviderConfigError(f"headlines must be one of {HEADLINE_PROVIDERS}")
        doc = None
        if documents == "sec-edgar":
            ua = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
            if not ua:
                raise ProviderConfigError(
                    "SEC EDGAR requires a contact User-Agent: set SEC_EDGAR_USER_AGENT="
                    "'Your Name your@email' (https://www.sec.gov/os/accessing-edgar-data)")
            doc = SecEdgarPressReleases(JsonHttp("sec", None, "", raw_cache_dir,
                                                 min_interval=0.12,
                                                 headers={"User-Agent": ua}))
        elif documents == "fmp":
            key = os.environ.get("FMP_API_KEY", "").strip()
            if not key:
                raise ProviderConfigError("documents=fmp requires FMP_API_KEY")
            doc = FMPTranscripts(JsonHttp("fmp", "apikey", key, raw_cache_dir, min_interval=0.25))
        news = None
        if headlines == "finnhub":
            key = os.environ.get("FINNHUB_API_KEY", "").strip()
            if not key:
                raise ProviderConfigError("headlines=finnhub requires FINNHUB_API_KEY")
            news = FinnhubNews(JsonHttp("finnhub", "token", key, raw_cache_dir,
                                        min_interval=1.05))
        return cls(YahooMarketData(), doc, news)
