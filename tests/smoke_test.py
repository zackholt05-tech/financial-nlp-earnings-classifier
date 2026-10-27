"""Fast end-to-end smoke test (tiny BERT, synthetic data, CPU).

Validates the full wiring (data -> split -> embeddings -> head -> train -> eval),
plus the leakage guard and an overfit-one-batch sanity check, WITHOUT downloading
the 440MB FinBERT. It loads the pinned 2-layer `google/bert_uncased_L-2_H-128_A-2`
(~17MB) from the Hugging Face Hub, so it is marked `slow`.

Run:  python tests/smoke_test.py      or      python -m pytest -m slow
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earnings_classifier.config import NEGATIVE_CONTROL, POSITIVE_CONTROL, Config
from earnings_classifier.data_collection import (
    DataValidationError,
    assert_no_leakage,
    collect_events,
    temporal_split,
)
from earnings_classifier.timing import ET
from earnings_classifier.embeddings import compute_embeddings, make_embedding_dataset
from earnings_classifier.evaluate import compute_metrics, format_metrics
from earnings_classifier.model import EndToEndModel, FusionHead
from earnings_classifier.preprocessing import RawEventDataset, collate_raw, load_tokenizer
from earnings_classifier.train import (
    e2e_forward,
    frozen_forward,
    overfit_one_batch,
    predict,
    train_model,
)
from earnings_classifier.utils import resolve_device, set_seed

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
checks = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    checks.append(cond)


def _split(cfg):
    events, _ = collect_events(cfg, cache_dir=None)
    events, report = temporal_split(events, cfg)
    return [e for e in events if e.split != "purged"], report


def run_checks() -> list:
    checks.clear()
    set_seed(0)
    device = torch.device("cpu")  # smoke test stays on CPU for determinism/speed
    cfg = Config.tiny(n_synthetic_events=80, epochs=4, data_mode=NEGATIVE_CONTROL)

    print("\n1. Leakage guard")
    t0 = datetime(2023, 1, 10, 16, 0, tzinfo=ET)
    assert_no_leakage([t0 - timedelta(days=1), t0 - timedelta(hours=1)], t0)  # ok
    raised = False
    try:
        assert_no_leakage([t0 + timedelta(days=1)], t0)  # future headline -> must raise
    except DataValidationError:
        raised = True
    check("rejects post-cutoff headline", raised)

    print("\n2. Data collection + temporal split (negative control)")
    events, report = _split(cfg)
    n = len(events)
    check("event count", n + sum(report["purged"].values()) == 80, f"{n} used + purged")
    train_last = max(e.exit_day for e in events if e.split == "train")
    test_first = min(e.entry_day for e in events if e.split == "test")
    check("temporal split (train label windows end before test)",
          train_last < test_first, f"{train_last} < {test_first}")
    check("both classes present", len({e.label for e in events}) == 2)

    print("\n3. Embeddings (tiny BERT)")
    emb = compute_embeddings(events, cfg, device, cache_dir=None)
    H = emb["t_emb"].shape[1]
    check("embedding shapes", emb["t_emb"].shape == emb["n_emb"].shape == (n, H), f"H={H}")
    check("presence flags set", emb["t_present"].mean() > 0 and emb["n_present"].mean() > 0)

    print("\n4. FusionHead training (frozen path)")
    loaders = {
        s: DataLoader(make_embedding_dataset(emb, s), batch_size=cfg.batch_size,
                      shuffle=(s == "train"))
        for s in ("train", "val", "test")
    }
    train_labels = emb["labels"][emb["splits"] == "train"]
    head = FusionHead(cfg, embed_dim=H)
    head, info = train_model(head, loaders["train"], loaders["val"],
                             train_labels, cfg, device, frozen_forward)
    y, p = predict(head, loaders["test"], device, frozen_forward)
    m = compute_metrics(y, p)
    check("training produced finite probs", np.isfinite(p).all())
    check("metrics computed", "auc" in m and "accuracy" in m)
    print("       " + format_metrics("smoke/test", m))

    print("\n5. Overfit-one-batch sanity (head must be able to learn)")
    batch = next(iter(loaders["train"]))
    acc = overfit_one_batch(FusionHead(cfg, embed_dim=H), batch, cfg, device,
                            frozen_forward, steps=300)
    check("overfits a single batch", acc >= 0.85, f"train acc={acc:.3f}")

    print("\n6. POSITIVE CONTROL — signal-bearing text must beat chance")
    # The discriminating test: if the label is encoded in the text, a correct
    # document->encode->pool->fuse->head path MUST score well above 0.5.
    # A null result here means text signal is silently dropped somewhere.
    cfg_sig = Config.tiny(n_synthetic_events=120, epochs=12, data_mode=POSITIVE_CONTROL)
    sig_events, _ = _split(cfg_sig)
    long_tok = len(load_tokenizer(cfg_sig).encode(sig_events[0].document))
    check("signal document spans multiple chunks",
          long_tok > cfg_sig.document_chunk_len, f"{long_tok} tokens")
    sig_emb = compute_embeddings(sig_events, cfg_sig, device, cache_dir=None)
    sig_loaders = {
        s: DataLoader(make_embedding_dataset(sig_emb, s), batch_size=cfg_sig.batch_size,
                      shuffle=(s == "train"))
        for s in ("train", "val", "test")
    }
    sig_head = FusionHead(cfg_sig, embed_dim=sig_emb["t_emb"].shape[1])
    sig_head, _ = train_model(sig_head, sig_loaders["train"], sig_loaders["val"],
                              sig_emb["labels"][sig_emb["splits"] == "train"],
                              cfg_sig, device, frozen_forward)
    ys, ps = predict(sig_head, sig_loaders["test"], device, frozen_forward)
    sig_auc = compute_metrics(ys, ps)["auc"]
    check("signal-bearing test AUC >> chance", sig_auc > 0.75, f"AUC={sig_auc:.3f}")

    print("\n7. End-to-end raw path (forward + one step)")
    tokenizer = load_tokenizer(cfg)
    cfg_ft = Config.tiny(n_synthetic_events=80, epochs=1, freeze_encoder=False)
    ds = RawEventDataset([e for e in events if e.split == "train"], tokenizer, cfg_ft)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_raw)
    model = EndToEndModel(cfg_ft)
    rb = next(iter(loader))
    logits = e2e_forward(model, rb)
    check("e2e forward shape", logits.shape == (rb["label"].shape[0],), str(tuple(logits.shape)))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, rb["label"].float())
    loss.backward()
    check("e2e backward produces grads",
          any(p.grad is not None for p in model.parameters() if p.requires_grad))

    return list(checks)


@pytest.mark.slow
def test_smoke_end_to_end():
    results = run_checks()
    assert results and all(results), f"{sum(results)}/{len(results)} smoke checks passed"


def main() -> None:
    results = run_checks()
    print("\n" + "=" * 60)
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
        sys.exit(0)
    print(f"{sum(results)}/{len(results)} passed — FAILURES ABOVE")
    sys.exit(1)


if __name__ == "__main__":
    main()
