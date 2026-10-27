"""End-to-end pipeline: collect -> split -> embed -> train -> evaluate -> run directory.

Examples
--------
  # Pipeline checks (no keys/network beyond the model download):
  python main.py --data-mode synthetic-negative-control     # expect AUC ~0.5
  python main.py --data-mode synthetic-positive-control     # expect AUC >> 0.5

  # One-minute CPU demo with a tiny 2-layer BERT instead of FinBERT:
  python main.py --preset tiny --data-mode synthetic-positive-control --device cpu

  # Real data: yfinance prices/dates + SEC EDGAR earnings press releases
  # (8-K Item 2.02, Exhibit 99.1) + Finnhub headlines:
  SEC_EDGAR_USER_AGENT="Name email" FINNHUB_API_KEY=... python main.py --data-mode real
  # Collection + coverage report only (no model):
  python main.py --data-mode real --collect-only

Each run writes runs/<UTC timestamp>_<mode>_<fingerprint>/ with manifest.json,
metrics.json, predictions.csv and the trained head (weights are not committed).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from earnings_classifier.baselines import logistic_regression_baseline, majority_class_baseline
from earnings_classifier.config import (
    DATA_MODES,
    DOCUMENT_PROVIDERS,
    HEADLINE_PROVIDERS,
    NEGATIVE_CONTROL,
    POSITIVE_CONTROL,
    REAL,
    Config,
)
from earnings_classifier.data_collection import (
    collect_events,
    dataset_fingerprint,
    load_frozen_events,
    temporal_split,
    validate_events,
)
from earnings_classifier.embeddings import (
    compute_embeddings,
    make_embedding_dataset,
    split_concat_features,
)
from earnings_classifier.evaluate import compute_metrics, format_metrics
from earnings_classifier.model import EndToEndModel, FusionHead
from earnings_classifier.preprocessing import RawEventDataset, collate_raw, load_tokenizer
from earnings_classifier.provenance import (
    build_manifest,
    create_run_dir,
    dependency_versions,
    git_info,
    platform_info,
    write_json,
    write_predictions,
)
from earnings_classifier.train import e2e_forward, frozen_forward, predict, train_model
from earnings_classifier.utils import get_logger, resolve_device, set_seed

log = get_logger()

BANNERS = {
    REAL: "REAL MARKET DATA",
    NEGATIVE_CONTROL: "SYNTHETIC NEGATIVE CONTROL - pipeline check: labels are independent "
                      "of the text, so AUC ~0.5 is expected. Not market evidence.",
    POSITIVE_CONTROL: "SYNTHETIC POSITIVE CONTROL - pipeline check: the text encodes the "
                      "label, so AUC >> 0.5 is expected. Not market evidence.",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-earnings direction classifier")
    p.add_argument("--data-mode", choices=DATA_MODES, default=None,
                   help=f"Experiment mode (default: {REAL}).")
    p.add_argument("--synthetic", action="store_true",
                   help=f"Deprecated alias for --data-mode {NEGATIVE_CONTROL}.")
    p.add_argument("--synthetic-signal", action="store_true",
                   help=f"Deprecated alias for --data-mode {POSITIVE_CONTROL}.")
    p.add_argument("--preset", choices=("default", "tiny"), default="default",
                   help="'tiny' = 2-layer BERT + small windows (fast CPU demo/tests).")
    p.add_argument("--finetune", action="store_true",
                   help="Fine-tune the encoder end-to-end instead of a frozen-encoder head.")
    p.add_argument("--model", type=str, default="", help="HF model id for the encoder.")
    p.add_argument("--model-revision", type=str, default="",
                   help="Full commit hash for --model (required with --model).")
    p.add_argument("--modalities", choices=("both", "document", "headlines"), default="both",
                   help="Ablation: which text streams the classifiers may use.")
    p.add_argument("--documents", choices=DOCUMENT_PROVIDERS, default=None,
                   help="Real mode: earnings-document provider (default: sec-edgar press "
                        "releases; fmp = call transcripts, paid plan).")
    p.add_argument("--headlines", choices=HEADLINE_PROVIDERS, default=None,
                   help="Real mode: headline provider (default: finnhub).")
    p.add_argument("--tickers", nargs="*", default=None, help="Ticker universe.")
    p.add_argument("--max-tickers", type=int, default=0, help="Cap the universe.")
    p.add_argument("--earnings-per-ticker", type=int, default=0,
                   help="Real mode: most recent announcements per ticker.")
    p.add_argument("--frozen-events", type=str, default="",
                   help="Real mode: analyse this frozen events file instead of collecting "
                        "(requires --expect-fingerprint; no provider access).")
    p.add_argument("--expect-fingerprint", type=str, default="",
                   help="Dataset fingerprint the events must have; the run aborts otherwise.")
    p.add_argument("--val-start-date", type=str, default="",
                   help="Explicit first validation entry date (YYYY-MM-DD).")
    p.add_argument("--test-start-date", type=str, default="",
                   help="Explicit first test entry date (YYYY-MM-DD).")
    p.add_argument("--collect-only", action="store_true",
                   help="Collect, validate and split the events, print the coverage report, "
                        "and stop (no model, no run directory).")
    p.add_argument("--n-synthetic", type=int, default=0, help="Synthetic event count.")
    p.add_argument("--epochs", type=int, default=0, help="Override epoch count.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="", help="cpu | mps | cuda")
    p.add_argument("--runs-dir", type=str, default="", help="Parent directory for run outputs.")
    cache = p.add_mutually_exclusive_group()
    cache.add_argument("--no-cache", action="store_true",
                       help="Do not read or write any cache (events, embeddings, raw responses).")
    cache.add_argument("--refresh-cache", action="store_true",
                       help="Ignore existing caches, recompute, and overwrite them.")
    args = p.parse_args(argv)

    aliases = [m for m, on in ((NEGATIVE_CONTROL, args.synthetic),
                               (POSITIVE_CONTROL, args.synthetic_signal)) if on]
    if args.synthetic and args.synthetic_signal:
        aliases = [POSITIVE_CONTROL]  # pre-audit behaviour: --synthetic-signal implied --synthetic
    if aliases:
        if args.data_mode and args.data_mode != aliases[0]:
            p.error(f"--data-mode {args.data_mode} conflicts with the deprecated alias flag")
        log.warning("--synthetic/--synthetic-signal are deprecated; use --data-mode %s", aliases[0])
        args.data_mode = aliases[0]
    args.data_mode = args.data_mode or REAL
    if bool(args.model) != bool(args.model_revision):
        p.error("--model and --model-revision must be given together")
    if args.frozen_events and not args.expect_fingerprint:
        p.error("--frozen-events requires --expect-fingerprint")
    if args.frozen_events and args.data_mode != REAL:
        p.error("--frozen-events is for --data-mode real")
    return args


def build_config(args: argparse.Namespace) -> Config:
    base = Config.tiny() if args.preset == "tiny" else Config()
    over: Dict = {"data_mode": args.data_mode, "seed": args.seed,
                  "freeze_encoder": not args.finetune}
    if args.model:
        over.update(model_name=args.model, model_revision=args.model_revision)
    if args.tickers:
        over["tickers"] = list(args.tickers)
    if args.max_tickers:
        over["tickers"] = list(over.get("tickers", base.tickers))[: args.max_tickers]
    if args.documents:
        over["document_provider"] = args.documents
    if args.headlines:
        over["headline_provider"] = args.headlines
    if args.earnings_per_ticker:
        over["earnings_per_ticker"] = args.earnings_per_ticker
    if args.val_start_date:
        over["val_start_date"] = args.val_start_date
    if args.test_start_date:
        over["test_start_date"] = args.test_start_date
    if args.epochs:
        over["epochs"] = args.epochs
    if args.n_synthetic:
        over["n_synthetic_events"] = args.n_synthetic
    if args.device:
        over["device"] = args.device
    if args.runs_dir:
        over["runs_dir"] = args.runs_dir
    return dataclasses.replace(base, **over)


def _loaders(emb: Dict, cfg: Config, modalities: str) -> Dict[str, DataLoader]:
    return {
        s: DataLoader(make_embedding_dataset(emb, s, modalities), batch_size=cfg.batch_size,
                      shuffle=(s == "train"))
        for s in ("train", "val", "test")
    }


def run_frozen(cfg: Config, emb: Dict, device, modalities: str):
    embed_dim = emb["t_emb"].shape[1]
    loaders = _loaders(emb, cfg, modalities)
    train_labels = emb["labels"][emb["splits"] == "train"]
    model = FusionHead(cfg, embed_dim=embed_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("FusionHead trainable params: %d", n_params)
    model, info = train_model(model, loaders["train"], loaders["val"], train_labels, cfg,
                              device, frozen_forward)
    probs = {s: predict(model, loaders[s], device, frozen_forward)[1] for s in ("val", "test")}
    return model, info, n_params, probs


def run_end_to_end(cfg: Config, events, device):
    tokenizer = load_tokenizer(cfg)
    by_split = {s: [e for e in events if e.split == s] for s in ("train", "val", "test")}
    loaders = {
        s: DataLoader(RawEventDataset(evs, tokenizer, cfg), batch_size=cfg.batch_size,
                      shuffle=(s == "train"), collate_fn=collate_raw)
        for s, evs in by_split.items()
    }
    train_labels = np.array([e.label for e in by_split["train"]])
    model = EndToEndModel(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("EndToEndModel trainable params: %d", n_params)
    model, info = train_model(model, loaders["train"], loaders["val"], train_labels, cfg,
                              device, e2e_forward)
    probs = {s: predict(model, loaders[s], device, e2e_forward)[1] for s in ("val", "test")}
    return model, info, n_params, probs


def main(argv: Optional[List[str]] = None) -> str:
    args = parse_args(argv)
    cfg = build_config(args)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    use_cache = not (args.no_cache or args.refresh_cache)
    write_cache = not args.no_cache
    log.info("Mode: %s | device: %s | encoder: %s@%s | %s", cfg.data_mode, device,
             cfg.model_name, cfg.model_revision[:7],
             "finetune" if not cfg.freeze_encoder else "frozen-encoder head")

    # ---- Data --------------------------------------------------------------
    if args.frozen_events:
        events, payload = load_frozen_events(args.frozen_events, args.expect_fingerprint, cfg)
        collect_info = {"frozen_events": args.frozen_events, "cache_key": payload["cache_key"],
                        "report": payload["report"], "created_at": payload["created_at"]}
    else:
        events, collect_info = collect_events(cfg, cache_dir=cfg.cache_dir,
                                              use_cache=use_cache, write_cache=write_cache)
    validate_events(events, cfg)
    fingerprint = dataset_fingerprint(events)
    if args.expect_fingerprint and fingerprint != args.expect_fingerprint:
        raise SystemExit(f"dataset fingerprint {fingerprint} != expected "
                         f"{args.expect_fingerprint}; aborting")
    events, split_report = temporal_split(events, cfg)
    used = [e for e in events if e.split in ("train", "val", "test")]
    if args.collect_only:
        print(json.dumps({"data_mode": cfg.data_mode, "dataset_fingerprint": fingerprint,
                          "events_cache": collect_info, "split": split_report},
                         indent=2, default=str))
        return ""

    # ---- Embeddings + models -------------------------------------------------
    emb = compute_embeddings(used, cfg, device, cache_dir=cfg.cache_dir, use_cache=use_cache,
                             write_cache=write_cache)
    if cfg.freeze_encoder:
        model, info, n_params, probs = run_frozen(cfg, emb, device, args.modalities)
    else:
        if args.modalities != "both":
            raise SystemExit("--modalities ablations are implemented for the frozen path only")
        model, info, n_params, probs = run_end_to_end(cfg, used, device)

    y = {s: emb["labels"][emb["splits"] == s] for s in ("train", "val", "test")}
    train_rate = float(np.mean(y["train"]))
    Xtr, ytr = split_concat_features(emb, "train", args.modalities)
    metrics: Dict = {"neural": {}, "logistic_regression": {}, "majority_class": {}}
    lr_probs = {}
    for s in ("val", "test"):
        metrics["neural"][s] = compute_metrics(y[s], probs[s], train_pos_rate=train_rate)
        Xs, ys = split_concat_features(emb, s, args.modalities)
        metrics["logistic_regression"][s], lr_probs[s] = logistic_regression_baseline(
            Xtr, ytr, Xs, ys, cfg.seed)
        metrics["majority_class"][s] = majority_class_baseline(y["train"], y[s])

    # ---- Report ----------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"POST-EARNINGS DIRECTION CLASSIFIER - {BANNERS[cfg.data_mode]}")
    print("=" * 78)
    print(f"events={len(events)} used={len(used)} purged={split_report['purged']} "
          f"modalities={args.modalities} encoder={cfg.model_name}@{cfg.model_revision[:7]}")
    for s in ("train", "val", "test"):
        r = split_report["splits"][s]
        print(f"  {s:5s} n={r['n']:4d} pos_rate={r['pos_rate']:.3f} "
              f"entries {r['first_entry']}..{r['last_entry']}")
    print("-" * 78)
    for name in ("majority_class", "logistic_regression", "neural"):
        for s in ("val", "test"):
            print(format_metrics(f"{name} / {s}", metrics[name][s]))
    print("=" * 78)
    print("Validation numbers were used for checkpoint selection; test numbers are held out.")
    print("Classification metrics are not trading performance.\n")

    # ---- Run directory -------------------------------------------------------
    run_dir = create_run_dir(cfg.runs_dir, cfg.data_mode, fingerprint)
    by_id = {e.event_id: e for e in used}
    rows = []
    for s in ("val", "test"):
        ids = emb["event_ids"][emb["splits"] == s]
        for eid, p_nn, p_lr in zip(ids, probs[s], lr_probs[s]):
            e = by_id[str(eid)]
            rows.append({"event_id": e.event_id, "ticker": e.ticker, "split": s,
                         "entry_date": e.entry_date, "label": e.label,
                         "forward_return": e.forward_return,
                         "p_neural": float(p_nn), "p_logistic_regression": float(p_lr)})
    write_predictions(os.path.join(run_dir, "predictions.csv"), rows)
    write_json(os.path.join(run_dir, "metrics.json"),
               {"data_mode": cfg.data_mode, "modalities": args.modalities, "metrics": metrics,
                "train_info": info, "n_trainable_params": n_params})
    manifest = build_manifest(
        root=os.path.dirname(os.path.abspath(__file__)),
        data_mode=cfg.data_mode,
        is_control=cfg.is_synthetic,
        argv=sys.argv if argv is None else ["main.py", *argv],
        seed=cfg.seed,
        config=cfg.to_dict(),
        modalities=args.modalities,
        dataset_fingerprint=fingerprint,
        events_cache=collect_info,
        embeddings_cache={"key": emb["cache_key"], "hit": emb["cache_hit"],
                          "path": emb["cache_path"], "encode_device": emb["encode_device"]},
        cache_policy={"no_cache": args.no_cache, "refresh_cache": args.refresh_cache},
        model={"name": cfg.model_name, "revision": cfg.model_revision,
               "tokenizer": cfg.model_name, "tokenizer_revision": cfg.model_revision,
               "pooling": cfg.pooling, "freeze_encoder": cfg.freeze_encoder,
               "head_layers": cfg.head_layers, "n_trainable_params": n_params},
        counts={"events": len(events), "used": len(used), "split": split_report},
        dependencies=dependency_versions(),
        platform=platform_info(str(device)),
        git=git_info(os.path.dirname(os.path.abspath(__file__))),
    )
    write_json(os.path.join(run_dir, "manifest.json"), manifest)
    torch.save(model.state_dict(),
               os.path.join(run_dir, "head.pt" if cfg.freeze_encoder else "end_to_end.pt"))
    log.info("Saved run to %s/", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
