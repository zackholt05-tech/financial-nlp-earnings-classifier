"""Post-earnings stock direction classifier.

A dual-stream FinBERT model that fuses an earnings-document embedding (SEC
earnings press release by default, or an earnings-call transcript) with a
financial-news-headline embedding to predict 3-day post-earnings price
direction (up vs. down).
"""

__version__ = "0.1.0"
