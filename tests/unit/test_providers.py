"""Provider HTTP behaviour with a mocked opener: errors are loud, keys never leak."""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import date, datetime, timezone

import pytest

from earnings_classifier.providers import (
    FinnhubNews,
    JsonHttp,
    ProviderAuthError,
    ProviderConfigError,
    ProviderError,
    ProviderRateLimitError,
    Providers,
)

KEY = "sk-TEST-SECRET-123"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _opener(body=None, code=None, exc=None, seen=None):
    def open_(req, timeout):
        url = req.full_url
        if seen is not None:
            seen.append(url)
        if exc is not None:
            raise exc
        if code is not None:
            raise urllib.error.HTTPError(url, code, f"HTTP {code} for {url}", {}, None)
        return _Resp(json.dumps(body).encode())
    return open_


def _http(tmp_path=None, **kw):
    return JsonHttp("fmp", "apikey", KEY, raw_cache_dir=str(tmp_path) if tmp_path else None,
                    opener=_opener(**kw))


@pytest.mark.parametrize("code,exc", [(401, ProviderAuthError), (402, ProviderAuthError),
                                      (403, ProviderAuthError), (429, ProviderRateLimitError),
                                      (500, ProviderError)])
def test_http_errors_raise_without_leaking_key(code, exc):
    with pytest.raises(exc) as err:
        _http(code=code).get("https://example.test/x", {"symbol": "ACME"})
    assert KEY not in str(err.value)


def test_network_error_message_is_redacted():
    http = JsonHttp("fmp", "apikey", KEY,
                    opener=_opener(exc=urllib.error.URLError(f"bad host in ?apikey={KEY}")))
    with pytest.raises(ProviderError) as err:
        http.get("https://example.test/x", {"symbol": "ACME"})
    assert KEY not in str(err.value) and "***" in str(err.value)


def test_404_means_no_data():
    assert _http(code=404).get("https://example.test/x", {}) is None


def test_fmp_error_payload_is_an_auth_error():
    with pytest.raises(ProviderAuthError, match="refused"):
        _http(body={"Error Message": f"Invalid API KEY {KEY}"}).get("https://example.test/x", {})


def test_non_json_is_an_error():
    http = JsonHttp("fmp", "apikey", KEY, opener=lambda req, timeout: _Resp(b"<html>"))
    with pytest.raises(ProviderError, match="non-JSON"):
        http.get("https://example.test/x", {})


def test_raw_cache_is_keyed_without_secret_and_reused(tmp_path):
    seen = []
    http = JsonHttp("fmp", "apikey", KEY, raw_cache_dir=str(tmp_path),
                    opener=_opener(body=[{"a": 1}], seen=seen))
    assert http.get("https://example.test/x", {"symbol": "ACME"}) == [{"a": 1}]
    assert http.get("https://example.test/x", {"symbol": "ACME"}) == [{"a": 1}]
    assert len(seen) == 1 and KEY in seen[0]          # key is sent to the provider ...
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1 and KEY not in files[0].read_text()   # ... but never stored


def test_finnhub_parsing_and_schema_check():
    body = [{"datetime": 1720540800, "headline": " ACME previews quarter ", "source": "wire"},
            {"datetime": 1720540900, "headline": ""}]
    news = FinnhubNews(_http(body=body))
    (h,) = news.company_news("ACME", date(2024, 7, 1), date(2024, 7, 10))
    assert h.text == "ACME previews quarter"
    assert h.published == datetime(2024, 7, 9, 16, 0, tzinfo=timezone.utc)
    with pytest.raises(ProviderError, match="datetime/headline"):
        FinnhubNews(_http(body=[{"headline": "x"}])).company_news("ACME", date(2024, 7, 1),
                                                                   date(2024, 7, 2))


def test_providers_read_keys_only_from_environment(monkeypatch):
    for var in ("FMP_API_KEY", "FINNHUB_API_KEY", "SEC_EDGAR_USER_AGENT"):
        monkeypatch.delenv(var, raising=False)
    p = Providers.from_env(raw_cache_dir=None, documents="none", headlines="none")
    assert p.documents is None and p.news is None
    with pytest.raises(ProviderConfigError, match="FINNHUB_API_KEY"):
        Providers.from_env(raw_cache_dir=None, documents="none")
    with pytest.raises(ProviderConfigError, match="FMP_API_KEY"):
        Providers.from_env(raw_cache_dir=None, documents="fmp", headlines="none")
    monkeypatch.setenv("FINNHUB_API_KEY", KEY)
    p = Providers.from_env(raw_cache_dir=None, documents="none")
    assert p.news is not None and p.news.http.secret == KEY and p.documents is None
    monkeypatch.setenv("FMP_API_KEY", KEY)
    p = Providers.from_env(raw_cache_dir=None, documents="fmp", headlines="none")
    assert p.documents.source == "fmp" and p.documents.http.secret == KEY
    with pytest.raises(ProviderConfigError, match="documents must be one of"):
        Providers.from_env(raw_cache_dir=None, documents="bogus")
