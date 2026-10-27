"""SEC EDGAR press-release provider on hand-made fixtures (fake issuer ACME, CIK 1)."""

from __future__ import annotations

import io
import json
from datetime import date, datetime, timezone

import pytest

from earnings_classifier.providers import (
    JsonHttp,
    ProviderConfigError,
    ProviderError,
    Providers,
    SecEdgarPressReleases,
    exhibit_href,
    html_to_text,
)
from earnings_classifier.timing import ET

UA = "Test Person test@example.invalid"
SEC = "https://www.sec.gov"
SUB = "https://data.sec.gov/submissions"

TICKERS = {"0": {"cik_str": 1, "ticker": "ACME", "title": "ACME CORP"},
           "1": {"cik_str": 2, "ticker": "BRK-B", "title": "BERKSHIRE"}}


def _block(rows):
    keys = ("accessionNumber", "acceptanceDateTime", "form", "items")
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


RECENT = _block([
    ("0000000001-24-000300", "2024-10-31T20:30:05.000Z", "8-K", "2.02,9.01"),   # 16:30 EDT
    ("0000000001-24-000250", "2024-08-01T12:00:00.000Z", "10-Q", ""),
    ("0000000001-24-000200", "2024-07-11T03:10:00.000Z", "8-K", "2.02,9.01"),   # 23:10 EDT Jul 10
    ("0000000001-24-000150", "2024-05-02T20:30:00.000Z", "8-K", "5.02"),
    ("0000000001-24-000140", "2024-04-25T20:30:00.000Z", "8-K/A", "2.02"),
    ("0000000001-24-000130", "2024-04-24T20:30:00.000Z", "8-K", "7.01,12.02"),
])
OLDER = _block([("0000000001-14-000050", "2014-01-28T21:30:00.000Z", "8-K", "2.02,9.01")])  # EST
SUBMISSIONS = {"cik": "1", "filings": {"recent": RECENT, "files": [
    {"name": "CIK0000000001-submissions-001.json", "filingCount": 1,
     "filingFrom": "2010-01-01", "filingTo": "2015-12-31"}]}}

INDEX = """<table class="tableFile" summary="Document Format Files">
<tr><th scope="col">Seq</th><th scope="col">Description</th><th scope="col">Document</th>
<th scope="col">Type</th><th scope="col">Size</th></tr>
<tr><td scope="row">1</td><td scope="row">8-K</td>
<td scope="row"><a href="/ix?doc=/Archives/edgar/data/1/000000000124000200/acme-8k.htm">acme-8k.htm</a></td>
<td scope="row">8-K</td><td scope="row">30000</td></tr>
<tr class="evenRow"><td scope="row">2</td><td scope="row">CFO COMMENTARY</td>
<td scope="row"><a href="/Archives/edgar/data/1/000000000124000200/ex992.htm">ex992.htm</a></td>
<td scope="row">EX-99.2</td><td scope="row">100</td></tr>
<tr><td scope="row">3</td><td scope="row">PRESS RELEASE</td>
<td scope="row"><a href="/Archives/edgar/data/1/000000000124000200/ex991.htm">ex991.htm</a></td>
<td scope="row">EX-99.1</td><td scope="row">100</td></tr>
</table>"""
EXHIBIT = """<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>3
<FILENAME>ex991.htm
<DESCRIPTION>PRESS RELEASE
<TEXT>
<html><head><style>p {color: red}</style></head><body>
<p>ACME<b>Reports</b> Record Third&nbsp;Quarter &amp; Raises Outlook</p>
<table><tr><td>Revenue</td><td>$1,000</td></tr></table><script>var x = 1;</script>
</body></html>
</TEXT>
</DOCUMENT>"""


class FakeHttp:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params):
        self.calls.append(url)
        return self.routes.get(url)

    def get_text(self, url):
        self.calls.append(url)
        return self.routes.get(url)


def _sec(**extra):
    routes = {f"{SEC}/files/company_tickers.json": TICKERS,
              f"{SUB}/CIK0000000001.json": SUBMISSIONS,
              f"{SUB}/CIK0000000001-submissions-001.json": OLDER}
    routes.update(extra)
    return SecEdgarPressReleases(FakeHttp(routes))


def test_lists_only_original_8k_item_202_filings_with_et_dates():
    sec = _sec()
    refs = sec.list_documents("ACME", since=date(2024, 1, 1))
    assert [r.key for r in refs] == ["0000000001-24-000200", "0000000001-24-000300"]
    jul, oct_ = refs
    assert jul.doc_date == date(2024, 7, 10)          # 03:10Z on the 11th is 23:10 ET on the 10th
    assert jul.doc_time == datetime(2024, 7, 11, 3, 10, tzinfo=timezone.utc)
    assert oct_.doc_time.astimezone(ET).hour == 16 and oct_.doc_date == date(2024, 10, 31)
    assert jul.url == (f"{SEC}/Archives/edgar/data/1/000000000124000200/"
                       "0000000001-24-000200-index.htm")
    assert jul.fiscal_year is None and jul.fiscal_quarter is None
    # The older submissions page ends before `since`, so it is never requested.
    assert f"{SUB}/CIK0000000001-submissions-001.json" not in sec.http.calls


def test_older_submission_pages_are_read_when_needed():
    refs = _sec().list_documents("ACME", since=date(2014, 1, 1))
    assert refs[0].key == "0000000001-14-000050"
    assert refs[0].doc_time.astimezone(ET).hour == 16 and refs[0].doc_date == date(2014, 1, 28)


def test_cik_lookup_normalizes_class_shares_and_rejects_unknown_tickers():
    sec = _sec()
    assert sec.cik("acme") == 1 and sec.cik("BRK.B") == 2
    with pytest.raises(ProviderError, match="no CIK"):
        sec.cik("ZZZZ")


def test_submissions_schema_mismatch_is_an_error():
    sec = _sec(**{f"{SUB}/CIK0000000001.json": {"filings": {"recent": {"form": []}}}})
    with pytest.raises(ProviderError, match="expected columns"):
        sec.list_documents("ACME")


def test_fetch_prefers_ex_99_1_and_extracts_visible_text():
    sec = _sec(**{
        f"{SEC}/Archives/edgar/data/1/000000000124000200/0000000001-24-000200-index.htm": INDEX,
        f"{SEC}/Archives/edgar/data/1/000000000124000200/ex991.htm": EXHIBIT,
    })
    ref = [r for r in sec.list_documents("ACME") if r.key == "0000000001-24-000200"][0]
    text = sec.fetch_text("ACME", ref)
    assert text == "ACME Reports Record Third Quarter & Raises Outlook Revenue $1,000"
    assert sec.exhibit_types == {"EX-99.1": 1}


def test_filing_without_press_release_exhibit_has_no_text():
    no_ex = INDEX.replace("EX-99.1", "EX-10.1").replace("EX-99.2", "EX-10.2") \
        .replace("PRESS RELEASE", "CREDIT AGREEMENT")
    sec = _sec(**{f"{SEC}/Archives/edgar/data/1/000000000124000200/"
                  "0000000001-24-000200-index.htm": no_ex})
    ref = [r for r in sec.list_documents("ACME") if r.key == "0000000001-24-000200"][0]
    assert sec.fetch_text("ACME", ref) == ""
    assert sec.exhibit_types == {"none": 1}


def test_missing_index_is_an_error_not_missing_data():
    sec = _sec()
    ref = sec.list_documents("ACME")[0]
    with pytest.raises(ProviderError, match="index not found"):
        sec.fetch_text("ACME", ref)


def test_exhibit_href_falls_back_to_plain_ex_99():
    idx = INDEX.replace("EX-99.1", "EX-99").replace("EX-99.2", "EX-10.1")
    assert exhibit_href(idx) == ("EX-99", "/Archives/edgar/data/1/000000000124000200/ex991.htm")
    assert exhibit_href("<table></table>") is None


def test_mislabelled_release_falls_back_to_first_ex_99_exhibit():
    # Johnson & Johnson 2016-2019 style: release typed "EX-99.15", tables "EX-99.2O".
    idx = INDEX.replace("EX-99.2", "EX-99.15").replace("EX-99.1<", "EX-99.2O<") \
        .replace("CFO COMMENTARY", "EXHIBIT 99.15").replace("PRESS RELEASE", "EXHIBIT 99.2O")
    doc_type, link = exhibit_href(idx)
    assert (doc_type, link.rsplit("/", 1)[1]) == ("EX-99.15", "ex992.htm")   # first in order


def test_release_named_in_description_beats_other_ex_99_exhibits():
    # NVIDIA 2019-11-14 style: release typed "EX-95.1", CFO commentary typed "EX-99.2".
    idx = INDEX.replace("EX-99.1<", "EX-95.1<")
    doc_type, link = exhibit_href(idx)
    assert (doc_type, link.rsplit("/", 1)[1]) == ("EX-95.1", "ex991.htm")
    # Without a matching description, the first EX-99.* exhibit is used.
    assert exhibit_href(INDEX.replace("EX-99.1<", "EX-10.1<").replace("PRESS RELEASE", "MISC")) \
        == ("EX-99.2", "/Archives/edgar/data/1/000000000124000200/ex992.htm")


def test_html_to_text_handles_xml_declaration_and_plain_text():
    assert html_to_text('<?xml version="1.0" encoding="utf-8"?><html><head><title>Document'
                        '</title></head><p>A</p><p>B</p></html>') == "A B"
    assert html_to_text("<TYPE>EX-99\n<TEXT>\nPlain   text\nrelease\n</TEXT>") == \
        "Plain text release"


# ---- HTTP layer + environment ------------------------------------------------------
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_user_agent_is_sent_but_never_cached(tmp_path):
    seen = []

    def opener(req, timeout):
        seen.append(req)
        return _Resp(json.dumps(TICKERS).encode())

    http = JsonHttp("sec", None, "", raw_cache_dir=str(tmp_path), opener=opener,
                    headers={"User-Agent": UA})
    sec = SecEdgarPressReleases(http)
    assert sec.cik("ACME") == 1
    assert seen[0].get_header("User-agent") == UA
    assert seen[0].full_url == f"{SEC}/files/company_tickers.json"     # no secret parameter
    (f,) = tmp_path.rglob("*.json")
    assert UA not in f.read_text()


def test_sec_provider_requires_contact_user_agent(monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    monkeypatch.setenv("FINNHUB_API_KEY", "fake-finnhub")
    with pytest.raises(ProviderConfigError, match="User-Agent"):
        Providers.from_env(raw_cache_dir=None)
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", UA)
    p = Providers.from_env(raw_cache_dir=None)
    assert isinstance(p.documents, SecEdgarPressReleases)
    assert p.documents.http.headers == {"User-Agent": UA} and p.documents.http.min_interval > 0.1


def test_yahoo_retries_transient_failures_and_empty_frames():
    import pandas as pd

    from earnings_classifier.providers import _with_retries

    waits = []
    replies = iter([TimeoutError("curl 28"), pd.DataFrame(), pd.DataFrame({"Open": [1.0]})])

    def fetch():
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r

    assert len(_with_retries(fetch, "x", sleep=waits.append)) == 1
    assert waits == [3.0, 6.0]
    with pytest.raises(ProviderError, match="after 2 attempts"):
        _with_retries(lambda: (_ for _ in ()).throw(OSError("reset")), "x", attempts=2,
                      sleep=waits.append)
    assert _with_retries(lambda: pd.DataFrame(), "x", attempts=2, sleep=waits.append).empty
