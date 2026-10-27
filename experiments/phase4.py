"""Phase 4 real-data benchmark, exactly as pre-registered in docs/PHASE4_PROTOCOL.md.

  python experiments/phase4.py select      # validation only -> results/phase4/selection.json
  git add results/phase4/selection.json && git commit ...
  python experiments/phase4.py test        # once; refuses unless the selection is committed
  python experiments/phase4.py headlines   # exploratory, training-free

Data come only from the frozen events files (fingerprint-checked); nothing is
collected. Embeddings are computed once (accelerator) and cached; the heads
train on CPU so seeds reproduce exactly. The select stage never computes a
test-split prediction.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earnings_classifier.baselines import (  # noqa: E402
    logistic_regression_baseline,
    majority_class_baseline,
    ticker_prior_baseline,
)
from earnings_classifier.config import REAL, Config  # noqa: E402
from earnings_classifier.data_collection import load_frozen_events, temporal_split  # noqa: E402
from earnings_classifier.embeddings import (  # noqa: E402
    compute_embeddings,
    make_embedding_dataset,
    split_concat_features,
)
from earnings_classifier.evaluate import compute_metrics  # noqa: E402
from earnings_classifier.model import FusionHead  # noqa: E402
from earnings_classifier.provenance import (  # noqa: E402
    dependency_versions,
    git_info,
    platform_info,
    write_json,
)
from earnings_classifier.selection import require_committed, select_best  # noqa: E402
from earnings_classifier.train import frozen_forward, predict, train_model  # noqa: E402
from earnings_classifier.uncertainty import entry_weeks, week_block_bootstrap  # noqa: E402
from earnings_classifier.utils import get_logger, resolve_device, set_seed  # noqa: E402
from earnings_classifier.zeroshot import sentiment_scores  # noqa: E402

log = get_logger()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "phase4")

# ---- Frozen by docs/PHASE4_PROTOCOL.md -------------------------------------------
DOC_ONLY = ("data/frozen/events-real-document-only.json",
            "2caa9d48aedd07abb76bead601c3ddec026f6493edbb9fcf429defbf69e98259")
WITH_HEADLINES = ("data/frozen/events-real-with-headlines.json",
                  "ce118a67e773bc89b867d1965dac2ae9242f008eb776c3dc9fcdc5b96f5adcfa")
VAL_START, TEST_START = "2023-04-27", "2024-10-31"
CHUNKS = (1, 8)
LR_C = (0.01, 0.1, 1.0)
HEAD_LAYERS = (0, 1)
SEEDS = (0, 1, 2, 3, 4)
SHRINKAGE = 10.0
ZERO_SHOT_CHUNKS = 8
N_BOOT, BOOT_SEED = 2000, 0
SPLITS = ("train", "val", "test")


def base_config(headlines: bool = False) -> Config:
    return Config(data_mode=REAL, headline_provider="finnhub" if headlines else "none",
                  val_start_date=VAL_START, test_start_date=TEST_START, seed=0)


def load(dataset=DOC_ONLY, headlines: bool = False):
    cfg = base_config(headlines)
    events, payload = load_frozen_events(os.path.join(ROOT, dataset[0]), dataset[1], cfg)
    events, split_report = temporal_split(events, cfg)
    used = [e for e in events if e.split in SPLITS]
    return cfg, used, split_report


def embeddings_for(used, cfg: Config, chunks: int, device) -> Dict:
    c = dataclasses.replace(cfg, max_document_chunks=chunks)
    return compute_embeddings(used, c, device, cache_dir=os.path.join(ROOT, cfg.cache_dir))


def rank_metrics(y, s) -> Dict:
    y, s = np.asarray(y), np.asarray(s)
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, s)),
            "average_precision": float(average_precision_score(y, s)),
            "note": "rank-only score; probability metrics not applicable"}


def run_lr(emb: Dict, C: float, eval_splits) -> Dict:
    Xtr, ytr = split_concat_features(emb, "train")
    out = {}
    for s in eval_splits:
        Xs, ys = split_concat_features(emb, s)
        m, p = logistic_regression_baseline(Xtr, ytr, Xs, ys, seed=0, C=C)
        out[s] = (m, p)
    return out


def run_head(emb: Dict, cfg: Config, head_layers: int, seed: int, eval_splits) -> Dict:
    cpu = torch.device("cpu")
    c = dataclasses.replace(cfg, head_layers=head_layers, seed=seed)
    set_seed(seed)
    needed = ["train", "val", *[s for s in eval_splits if s not in ("train", "val")]]
    loaders = {s: DataLoader(make_embedding_dataset(emb, s), batch_size=c.batch_size,
                             shuffle=(s == "train")) for s in needed}   # no test loader in select
    model = FusionHead(c, embed_dim=emb["t_emb"].shape[1])
    y_train = emb["labels"][emb["splits"] == "train"]
    model, info = train_model(model, loaders["train"], loaders["val"], y_train, c, cpu,
                              frozen_forward)
    out = {"info": {"best_epoch": info["best_epoch"], "best_val_auc": info["best_val_auc"]}}
    for s in eval_splits:
        _, p = predict(model, loaders[s], cpu, frozen_forward)
        out[s] = p
    return out


def split_arrays(used, split):
    sub = [e for e in used if e.split == split]
    return sub, np.array([e.label for e in sub]), [e.ticker for e in sub]


def provenance(device) -> Dict:
    return {"dependencies": dependency_versions(), "platform": platform_info(str(device)),
            "git": git_info(ROOT)}


# ---------------------------------------------------------------------------
def stage_select(device) -> None:
    cfg, used, split_report = load()
    path = os.path.join(OUT, "selection.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; selection is done once")
    _, y_tr, t_tr = split_arrays(used, "train")
    _, y_val, t_val = split_arrays(used, "val")

    lr_rows, head_rows, emb_keys = [], [], {}
    for chunks in CHUNKS:
        emb = embeddings_for(used, cfg, chunks, device)
        emb_keys[chunks] = emb["cache_key"]
        assert list(emb["event_ids"]) == [e.event_id for e in used]
        for C in LR_C:
            m, _ = run_lr(emb, C, ["val"])["val"]
            lr_rows.append({"max_document_chunks": chunks, "C": C, "val_auc": m["auc"],
                            "val": m})
            log.info("LR chunks=%d C=%g val AUC %.4f", chunks, C, m["auc"])
        for hl in HEAD_LAYERS:
            per_seed = []
            for seed in SEEDS:
                r = run_head(emb, cfg, hl, seed, ["val"])
                per_seed.append({"seed": seed, "val_auc": compute_metrics(y_val, r["val"])["auc"],
                                 **r["info"]})
            aucs = [s["val_auc"] for s in per_seed]
            head_rows.append({"max_document_chunks": chunks, "head_layers": hl,
                              "val_auc": float(np.mean(aucs)), "val_auc_sd": float(np.std(aucs)),
                              "per_seed": per_seed})
            log.info("head chunks=%d layers=%d mean val AUC %.4f", chunks, hl, np.mean(aucs))

    lr_best = select_best(lr_rows, "val_auc", ["max_document_chunks", "C"])
    head_best = select_best(head_rows, "val_auc", ["max_document_chunks", "head_layers"])
    tp_val, _ = ticker_prior_baseline(t_tr, y_tr, t_val, y_val, SHRINKAGE)
    zs_val = sentiment_scores([e for e in used if e.split == "val"],
                              dataclasses.replace(cfg, max_document_chunks=ZERO_SHOT_CHUNKS),
                              device, "document")
    write_json(path, {
        "stage": "select", "protocol": "docs/PHASE4_PROTOCOL.md",
        "dataset": {"path": DOC_ONLY[0], "fingerprint": DOC_ONLY[1]},
        "split": split_report, "embedding_cache_keys": emb_keys,
        "grids": {"logistic_regression": lr_rows, "fusion_head": head_rows},
        "selected": {
            "logistic_regression": {k: lr_best[k] for k in ("max_document_chunks", "C")},
            "fusion_head": {k: head_best[k] for k in ("max_document_chunks", "head_layers")},
        },
        "fixed_baselines_val": {
            "majority_class": majority_class_baseline(y_tr, y_val),
            "ticker_prior": tp_val,
            "zero_shot_finbert": rank_metrics(y_val, zs_val),
        },
        "provenance": provenance(device),
    })
    print(f"selected LR {lr_best['max_document_chunks']} chunks C={lr_best['C']} "
          f"(val AUC {lr_best['val_auc']:.4f}); head {head_best['max_document_chunks']} chunks "
          f"{head_best['head_layers']} layers (mean val AUC {head_best['val_auc']:.4f})")
    print(f"wrote {path}; commit it before running the test stage")


def stage_test(device) -> None:
    sel_path = os.path.join(OUT, "selection.json")
    commit = require_committed(os.path.relpath(sel_path, ROOT), ROOT)
    out_path = os.path.join(OUT, "test_results.json")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; the test set is evaluated once")
    import json

    sel = json.load(open(sel_path))["selected"]
    cfg, used, split_report = load()
    _, y_tr, t_tr = split_arrays(used, "train")
    evs = {s: split_arrays(used, s) for s in ("val", "test")}

    probs: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in evs}
    metrics: Dict[str, Dict] = {}
    for s, (_, y, t) in evs.items():
        maj = np.full(len(y), float(y_tr.mean()))
        probs[s]["majority_class"] = maj
        metrics.setdefault("majority_class", {})[s] = majority_class_baseline(y_tr, y)
        m, p = ticker_prior_baseline(t_tr, y_tr, t, y, SHRINKAGE)
        probs[s]["ticker_prior"] = p
        metrics.setdefault("ticker_prior", {})[s] = m

    lr = sel["logistic_regression"]
    emb = embeddings_for(used, cfg, lr["max_document_chunks"], device)
    for s, (m, p) in run_lr(emb, lr["C"], ["val", "test"]).items():
        probs[s]["logistic_regression"] = p
        metrics.setdefault("logistic_regression", {})[s] = m

    hd = sel["fusion_head"]
    emb = embeddings_for(used, cfg, hd["max_document_chunks"], device)
    runs = [run_head(emb, cfg, hd["head_layers"], seed, ["val", "test"]) for seed in SEEDS]
    per_seed = []
    for seed, r in zip(SEEDS, runs):
        per_seed.append({"seed": seed, **r["info"],
                         **{f"{s}_auc": compute_metrics(evs[s][1], r[s])["auc"] for s in evs}})
    for s, (_, y, _) in evs.items():
        p = np.mean([r[s] for r in runs], axis=0)
        probs[s]["fusion_head"] = p
        metrics.setdefault("fusion_head", {})[s] = compute_metrics(y, p,
                                                                   train_pos_rate=float(y_tr.mean()))

    zcfg = dataclasses.replace(cfg, max_document_chunks=ZERO_SHOT_CHUNKS)
    for s, (sub, y, _) in evs.items():
        z = sentiment_scores(sub, zcfg, device, "document")
        probs[s]["zero_shot_finbert"] = z
        metrics.setdefault("zero_shot_finbert", {})[s] = rank_metrics(y, z)

    sub, y, _ = evs["test"]
    pairs = [("fusion_head", "ticker_prior"), ("logistic_regression", "ticker_prior"),
             ("zero_shot_finbert", "ticker_prior"), ("fusion_head", "logistic_regression"),
             ("ticker_prior", "majority_class")]
    boot = week_block_bootstrap(y, probs["test"], entry_weeks([e.entry_date for e in sub]),
                                pairs, n_boot=N_BOOT, seed=BOOT_SEED)
    write_json(out_path, {
        "stage": "test", "selection_commit": commit, "selected": sel,
        "dataset": {"path": DOC_ONLY[0], "fingerprint": DOC_ONLY[1]}, "split": split_report,
        "metrics": metrics, "fusion_head_per_seed": per_seed, "bootstrap_test": boot,
        "provenance": provenance(device),
    })
    rows = []
    for s, (sub_s, y_s, _) in evs.items():
        for e, i in zip(sub_s, range(len(sub_s))):
            rows.append({"event_id": e.event_id, "split": s, "entry_date": e.entry_date,
                         "label": e.label, **{k: float(v[i]) for k, v in probs[s].items()}})
    from earnings_classifier.provenance import write_predictions

    os.makedirs(os.path.join(ROOT, "runs", "phase4"), exist_ok=True)
    write_predictions(os.path.join(ROOT, "runs", "phase4", "test_stage_predictions.csv"), rows)
    print_summary(metrics, boot)


def stage_headlines(device) -> None:
    out_path = os.path.join(OUT, "exploratory_headlines.json")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists")
    cfg, used, split_report = load(WITH_HEADLINES, headlines=True)
    sub = [e for e in used if e.has_headlines]
    y = np.array([e.label for e in sub])
    weeks = entry_weeks([e.entry_date for e in sub])
    zcfg = dataclasses.replace(cfg, max_document_chunks=ZERO_SHOT_CHUNKS)
    h = sentiment_scores(sub, zcfg, device, "headlines")
    both = np.array([e.has_document for e in sub])
    d = sentiment_scores([e for e in sub if e.has_document], zcfg, device, "document")
    res = {
        "stage": "exploratory_headlines", "label": "EXPLORATORY - no training, no selection",
        "dataset": {"path": WITH_HEADLINES[0], "fingerprint": WITH_HEADLINES[1]},
        "events_with_headlines": int(len(sub)),
        "entry_range": [min(e.entry_date for e in sub), max(e.entry_date for e in sub)],
        "pos_rate": float(y.mean()),
        "headlines_n_per_event": {"median": float(np.median([len(e.headlines) for e in sub])),
                                  "min": int(min(len(e.headlines) for e in sub)),
                                  "max": int(max(len(e.headlines) for e in sub))},
        "all_with_headlines": {
            "zero_shot_headlines": rank_metrics(y, h),
            "bootstrap": week_block_bootstrap(y, {"zero_shot_headlines": h}, weeks,
                                              n_boot=N_BOOT, seed=BOOT_SEED),
        },
        "with_headlines_and_document": {
            "n": int(both.sum()),
            "zero_shot_headlines": rank_metrics(y[both], h[both]),
            "zero_shot_document": rank_metrics(y[both], d),
            "bootstrap": week_block_bootstrap(
                y[both], {"zero_shot_headlines": h[both], "zero_shot_document": d},
                weeks[both], [("zero_shot_headlines", "zero_shot_document")],
                n_boot=N_BOOT, seed=BOOT_SEED),
        },
        "provenance": provenance(device),
    }
    write_json(out_path, res)
    b = res["all_with_headlines"]["bootstrap"]["auc"]["zero_shot_headlines"]
    print(f"EXPLORATORY headlines zero-shot AUC {b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] "
          f"on {len(sub)} events")


def print_summary(metrics: Dict, boot: Dict) -> None:
    print("\nREAL DATA - held-out test (entries 2024-10-31..), evaluated once")
    print(f"{'model':22s} {'val AUC':>8s} {'test AUC':>9s}  95% week-block CI")
    for k in ("majority_class", "ticker_prior", "logistic_regression", "fusion_head",
              "zero_shot_finbert"):
        b = boot["auc"][k]
        print(f"{k:22s} {metrics[k]['val']['auc']:8.3f} {metrics[k]['test']['auc']:9.3f}  "
              f"[{b['lo']:.3f}, {b['hi']:.3f}]")
    for k, d in boot["diff"].items():
        print(f"  {k:40s} {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
    print(f"({boot['n_events']} events in {boot['n_weeks']} entry weeks; {boot['n_boot']} "
          f"resamples, {boot['n_skipped_single_class']} single-class skipped)")
    print("Classification metrics are not trading performance.")


def main(argv: List[str] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("select", "test", "headlines"))
    p.add_argument("--device", default="auto", help="embedding/zero-shot device")
    args = p.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    device = resolve_device(args.device)
    {"select": stage_select, "test": stage_test, "headlines": stage_headlines}[args.stage](device)


if __name__ == "__main__":
    main()
