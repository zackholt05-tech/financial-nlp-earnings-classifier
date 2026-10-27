# Legacy run (2026-06-18): synthetic positive control

These are byte-identical copies of the pre-audit `artifacts/` and `data/cache/` (sha256 verified 2026-10-27). The originals were left in place.

| File | sha256 |
|---|---|
| artifacts/config.json | 9d500a8fb26b17accdd24acd402a8f3fb6a02c314b9b248be6646ba512f24454 |
| artifacts/metrics.json | e2c5f1c48463b002e73ed2db4f0f912e6426256f4001b8e51e5f50763d204494 |
| artifacts/fusion_head.pt | 45326c4c4f1e0bcacf792849001e6fe6c179c3a7562e450c6e1387c70a1c97a6 |
| data_cache/events.json | 9c3e721d9d45a05778b3e0c6b49f8b1ebf6de1fbeff3ef7e8b56d2d2d748ffe9 |
| data_cache/embeddings.npz | 8571c438dd5221ebcce7381dcce2264a7bee1088633288cc2ec18a07e395ba4d |

## What this run is
- `config.json` has `synthetic=true` and `synthetic_signal=true`, so this is a **positive control**: the synthetic text deliberately encodes the label. The perfect val/test AUC of 1.0 is a pipeline check. It is **not** evidence of real-market predictive performance.
- The encoder was `ProsusAI/finbert` (frozen), with the default near-linear head (3,075 trainable parameters).

## Provenance gaps
- **No dataset fingerprint**, no dependency record, no git commit, and no command line were saved.
- **Unknown FinBERT snapshot.** Two snapshots are in the local HF cache: `4556d13…` (refs/main, `pytorch_model.bin`) and `7db323f…` (`model.safetensors`). Which one was loaded is not recorded. This does not affect the numbers: all 201 tensors in the two files are bit-identical (`torch.equal`, checked 2026-10-27).

## The two cache files do not describe the same data (verified 2026-10-27)
- `data_cache/events.json` matches the **negative-control** generator exactly (`Config(synthetic=True)`, seed 42, n=400): same transcripts, same labels.
- `data_cache/embeddings.npz` holds labels from the **positive-control** generator (`synthetic_signal=True`, seed 42). Its labels do **not** match `events.json`. Its split array matches only because both have 400 events.
- **Consequence.** With the pre-fix code, running `python main.py --synthetic` (the negative control) would:
  1. load the negative-control events from the event cache, and
  2. accept the positive-control embeddings and labels from the embedding cache, because the check only compares model name and event count.

  The "negative control" would then report positive-control scores. This is the concrete, observed instance of cache bugs (1) and (2) in `docs/STATUS.md`.

Use these files only as a historical record. They are not a reproducible experiment.
