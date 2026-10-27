"""Summarize the FinBERT synthetic-control runs (pipeline checks, not market evidence).

  python experiments/summarize_controls.py runs/controls-finbert  # -> results/controls_finbert.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(runs_dir: str) -> None:
    rows = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        mpath, fpath = os.path.join(d, "manifest.json"), os.path.join(d, "metrics.json")
        if not (os.path.exists(mpath) and os.path.exists(fpath)):
            continue
        man, met = json.load(open(mpath)), json.load(open(fpath))
        row = {"run": os.path.basename(d), "data_mode": man["data_mode"], "seed": man["seed"],
               "model": f"{man['model']['name']}@{man['model']['revision'][:7]}",
               "git_commit": man["git"].get("commit"),
               "n": {s: man["counts"]["split"]["splits"][s]["n"] for s in ("train", "val", "test")}}
        for name in ("neural", "logistic_regression"):
            for s in ("val", "test"):
                row[f"{name}_{s}_auc"] = met["metrics"][name][s]["auc"]
        rows.append(row)
    summary = {}
    for mode in sorted({r["data_mode"] for r in rows}):
        sub = [r for r in rows if r["data_mode"] == mode]
        summary[mode] = {"n_runs": len(sub), "seeds": [r["seed"] for r in sub]}
        for k in ("neural_val_auc", "neural_test_auc", "logistic_regression_val_auc",
                  "logistic_regression_test_auc"):
            v = np.array([r[k] for r in sub], dtype=float)
            summary[mode][k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1
                                else float("nan"), "min": float(v.min()), "max": float(v.max())}
    out = os.path.join(ROOT, "results", "controls_finbert.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"label": "SYNTHETIC PIPELINE CONTROLS - not market evidence",
                   "summary": summary, "runs": rows}, f, indent=2)
    for mode, s in summary.items():
        print(f"{mode:28s} runs={s['n_runs']} neural test AUC {s['neural_test_auc']['mean']:.3f} "
              f"± {s['neural_test_auc']['sd']:.3f}  LR test AUC "
              f"{s['logistic_regression_test_auc']['mean']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "controls-finbert"))
