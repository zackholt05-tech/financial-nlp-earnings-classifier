"""Aggregate description of the frozen real datasets (counts only; no provider text).

  python experiments/dataset_summary.py      # -> results/dataset_summary.json
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earnings_classifier.preprocessing import load_tokenizer  # noqa: E402
from experiments.phase4 import DOC_ONLY, ROOT, WITH_HEADLINES, load  # noqa: E402


def main() -> None:
    cfg, used, split_report = load(DOC_ONLY)
    payload = json.load(open(os.path.join(ROOT, DOC_ONLY[0])))
    tok = load_tokenizer(cfg)
    body = cfg.document_chunk_len - 2
    n_tokens = np.array([len(tok.encode(e.document, add_special_tokens=False)) for e in used])
    chars = np.array([len(e.document) for e in used])

    years: dict = {}
    for e in used:
        y = e.entry_date[:4]
        years.setdefault(y, Counter())[e.split] += 1
        years[y]["up"] += e.label
    per_ticker = Counter(e.ticker for e in used)

    _, used_h, split_h = load(WITH_HEADLINES, headlines=True)
    with_h = [e for e in used_h if e.has_headlines]
    out = {
        "label": "Aggregates of the frozen real datasets (not redistributed)",
        "document_only": {
            "fingerprint": DOC_ONLY[1],
            "collection_report": {k: payload["report"][k] for k in
                                  ("tickers_requested", "announcements_seen", "usable_events",
                                   "excluded", "document_missing", "document_exhibit_types",
                                   "errors")},
            "events_after_purge": len(used),
            "split": split_report,
            "sessions": dict(Counter(e.session for e in used)),
            "per_year": {y: dict(c) for y, c in sorted(years.items())},
            "per_ticker": {"n_tickers": len(per_ticker), "min": min(per_ticker.values()),
                           "median": float(np.median(list(per_ticker.values()))),
                           "max": max(per_ticker.values()), "counts": dict(sorted(per_ticker.items()))},
            "document_chars": {q: int(np.percentile(chars, p)) for q, p in
                               (("p10", 10), ("median", 50), ("p90", 90))},
            "document_tokens": {
                "tokenizer": f"{cfg.model_name}@{cfg.model_revision}",
                "median": int(np.median(n_tokens)), "p10": int(np.percentile(n_tokens, 10)),
                "p90": int(np.percentile(n_tokens, 90)),
                "chunk_body_tokens": body,
                "share_longer_than_1_chunk": float(np.mean(n_tokens > body)),
                "share_longer_than_8_chunks": float(np.mean(n_tokens > 8 * body)),
                "median_share_encoded_8_chunks": float(np.median(np.minimum(1, 8 * body / n_tokens))),
            },
        },
        "with_headlines": {
            "fingerprint": WITH_HEADLINES[1],
            "events_after_purge": len(used_h),
            "events_with_headlines": len(with_h),
            "with_headlines_by_split": dict(Counter(e.split for e in with_h)),
            "modality_by_split": {s: v["modality"] for s, v in split_h["splits"].items()},
            "first_entry_with_headlines": min(e.entry_date for e in with_h),
        },
    }
    path = os.path.join(ROOT, "results", "dataset_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    d = out["document_only"]
    print(f"{len(used)} events, {d['per_ticker']['n_tickers']} tickers; tokens median "
          f"{d['document_tokens']['median']}, >1 chunk {d['document_tokens']['share_longer_than_1_chunk']:.2f}, "
          f">8 chunks {d['document_tokens']['share_longer_than_8_chunks']:.2f}; sessions {d['sessions']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
