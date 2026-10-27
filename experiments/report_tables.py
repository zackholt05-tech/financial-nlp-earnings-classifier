"""Render the Phase 4 result files as Markdown tables (no numbers are typed by hand).

  python experiments/report_tables.py > results/phase4/tables.md
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
NAMES = {"majority_class": "Train-majority", "ticker_prior": "Ticker prior",
         "logistic_regression": "Logistic regression", "fusion_head": "FinBERT fusion head",
         "zero_shot_finbert": "Zero-shot FinBERT sentiment"}


def f(x, d=3):
    return "—" if x is None or x != x else f"{x:.{d}f}"


def main() -> None:
    sel = json.load(open(os.path.join(R, "phase4", "selection.json")))
    test = json.load(open(os.path.join(R, "phase4", "test_results.json")))
    expl = json.load(open(os.path.join(R, "phase4", "exploratory_headlines.json")))
    ctl = json.load(open(os.path.join(R, "controls_finbert.json")))

    print("### Data and split\n")
    print("| split | entries | n | up | up-rate |\n|---|---|---|---|---|")
    for s, v in sel["split"]["splits"].items():
        print(f"| {s} | {v['first_entry']} .. {v['last_entry']} | {v['n']} | {v['n_pos']} | "
              f"{f(v['pos_rate'])} |")
    p = sel["split"]["purged"]
    print(f"\nPurged (label window overlaps the next split): train {p['train']}, val {p['val']}.\n")

    print("### Validation grid (selection)\n")
    print("| model | chunks | setting | val AUC |\n|---|---|---|---|")
    for r in sel["grids"]["logistic_regression"]:
        print(f"| Logistic regression | {r['max_document_chunks']} | C = {r['C']} | "
              f"{f(r['val_auc'])} |")
    for r in sel["grids"]["fusion_head"]:
        print(f"| Fusion head | {r['max_document_chunks']} | {r['head_layers']} hidden layers | "
              f"{f(r['val_auc'])} ± {f(r['val_auc_sd'])} (5 seeds) |")
    s = sel["selected"]
    print(f"\nSelected: logistic regression {s['logistic_regression']}, fusion head "
          f"{s['fusion_head']}.\n")

    b = test["bootstrap_test"]
    print("### Held-out test (evaluated once)\n")
    print(f"{b['n_events']} events in {b['n_weeks']} entry weeks; 95% intervals from "
          f"{b['n_boot']} week-block bootstrap resamples.\n")
    print("| model | val AUC | test AUC | 95% CI | AP | Brier | log loss | accuracy | F1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, name in NAMES.items():
        m, v, ci = test["metrics"][k]["test"], test["metrics"][k]["val"], b["auc"][k]
        print(f"| {name} | {f(v['auc'])} | {f(m['auc'])} | [{f(ci['lo'])}, {f(ci['hi'])}] | "
              f"{f(m['average_precision'])} | {f(m.get('brier'))} | {f(m.get('log_loss'))} | "
              f"{f(m.get('accuracy'))} | {f(m.get('f1'))} |")
    print("\n| paired difference (test AUC) | estimate | 95% CI | share of resamples ≤ 0 |")
    print("|---|---|---|---|")
    for k, d in b["diff"].items():
        a, c = k.split(" - ")
        print(f"| {NAMES[a]} − {NAMES[c]} | {d['point']:+.3f} | [{d['lo']:+.3f}, {d['hi']:+.3f}] "
              f"| {f(d['frac_le_0'], 2)} |")
    print("\n| fusion-head seed | best epoch | val AUC | test AUC |\n|---|---|---|---|")
    for r in test["fusion_head_per_seed"]:
        print(f"| {r['seed']} | {r['best_epoch']} | {f(r['val_auc'])} | {f(r['test_auc'])} |")

    print("\n### Exploratory: headlines (no training, no selection)\n")
    print(f"{expl['events_with_headlines']} events with pre-cutoff headlines, entries "
          f"{expl['entry_range'][0]} .. {expl['entry_range'][1]}, up-rate {f(expl['pos_rate'])}.\n")
    print("| sample | score | n | AUC | 95% CI |\n|---|---|---|---|---|")
    a = expl["all_with_headlines"]["bootstrap"]
    ci = a["auc"]["zero_shot_headlines"]
    print(f"| all with headlines | headline sentiment | {a['n_events']} | {f(ci['point'])} | "
          f"[{f(ci['lo'])}, {f(ci['hi'])}] |")
    w = expl["with_headlines_and_document"]["bootstrap"]
    for k, lab in (("zero_shot_headlines", "headline sentiment"),
                   ("zero_shot_document", "release sentiment")):
        ci = w["auc"][k]
        print(f"| headlines and release | {lab} | {w['n_events']} | {f(ci['point'])} | "
              f"[{f(ci['lo'])}, {f(ci['hi'])}] |")
    d = w["diff"]["zero_shot_headlines - zero_shot_document"]
    print(f"\nHeadline − release sentiment AUC: {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}].\n")

    print("### Synthetic pipeline controls (FinBERT, seeds 0–4; not market evidence)\n")
    print("| control | runs | neural test AUC (mean ± sd) | range | LR test AUC (mean) |")
    print("|---|---|---|---|---|")
    for mode, v in ctl["summary"].items():
        n = v["neural_test_auc"]
        print(f"| {mode} | {v['n_runs']} | {f(n['mean'])} ± {f(n['sd'])} | "
              f"{f(n['min'])} .. {f(n['max'])} | {f(v['logistic_regression_test_auc']['mean'])} |")


if __name__ == "__main__":
    main()
