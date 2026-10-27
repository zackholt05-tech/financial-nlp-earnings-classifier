"""Generate every number, table and figure the paper uses from the saved results.

  python experiments/paper_assets.py     # -> paper/generated/{numbers.tex, tab_*.tex, fig_*.pdf}

The paper text uses only the macros in numbers.tex, so each reported value
traces to a file under results/.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "generated")
MODELS = [("majority_class", "Train-majority"), ("ticker_prior", "Ticker prior"),
          ("logistic_regression", "Logistic regression"), ("fusion_head", "FinBERT fusion head"),
          ("zero_shot_finbert", "Zero-shot FinBERT")]
MACRO = {"majority_class": "Maj", "ticker_prior": "Tick", "logistic_regression": "LR",
         "fusion_head": "Head", "zero_shot_finbert": "Zs"}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def load(*p):
    return json.load(open(os.path.join(RES, *p)))


def f3(x):
    return f"{x:.3f}"


def signed(x):
    return f"{x:+.3f}".replace("-", "$-$").replace("+", "$+$")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    sel, test = load("phase4", "selection.json"), load("phase4", "test_results.json")
    expl, ctl, ds = load("phase4", "exploratory_headlines.json"), load("controls_finbert.json"), \
        load("dataset_summary.json")
    boot = test["bootstrap_test"]
    d = ds["document_only"]
    rep = d["collection_report"]
    M: dict = {}

    # ---- data ------------------------------------------------------------------------
    M.update(nAnnouncements=rep["announcements_seen"], nUsable=rep["usable_events"],
             nEvents=d["events_after_purge"], nTickers=rep["tickers_requested"],
             nNoDocWindow=rep["document_missing"].get("no_document_in_window", 0),
             nAMC=d["sessions"].get("AMC", 0), nBMO=d["sessions"].get("BMO", 0),
             nDMH=d["sessions"].get("DMH", 0),
             tokMedian=f"{d['document_tokens']['median']:,}".replace(",", "{,}"),
             tokPten=f"{d['document_tokens']['p10']:,}".replace(",", "{,}"),
             tokPninety=f"{d['document_tokens']['p90']:,}".replace(",", "{,}"),
             shareOverEight=f"{100 * d['document_tokens']['share_longer_than_8_chunks']:.0f}",
             shareEncodedEight=f"{100 * d['document_tokens']['median_share_encoded_8_chunks']:.0f}",
             charMedian=f"{d['document_chars']['median']:,}".replace(",", "{,}"),
             nExhibitStd=rep["document_exhibit_types"].get("EX-99.1", 0),
             nHeadlineEvents=ds["with_headlines"]["events_with_headlines"],
             firstHeadlineEntry=ds["with_headlines"]["first_entry_with_headlines"])
    for s, v in sel["split"]["splits"].items():
        k = s.capitalize()
        M[f"n{k}"], M[f"up{k}"] = v["n"], f3(v["pos_rate"])
        M[f"first{k}"], M[f"last{k}"] = v["first_entry"], v["last_entry"]
    M["purgedTrain"], M["purgedVal"] = sel["split"]["purged"]["train"], sel["split"]["purged"]["val"]

    # ---- selection / test -----------------------------------------------------------
    s = sel["selected"]
    M.update(selLRChunks=s["logistic_regression"]["max_document_chunks"],
             selLRC=s["logistic_regression"]["C"],
             selHeadChunks=s["fusion_head"]["max_document_chunks"],
             selHeadLayers=s["fusion_head"]["head_layers"],
             nWeeks=boot["n_weeks"], nBoot=f"{boot['n_boot']:,}".replace(",", "{,}"))
    for key, _ in MODELS:
        m = MACRO[key]
        M[f"val{m}"] = f3(test["metrics"][key]["val"]["auc"])
        M[f"test{m}"] = f3(test["metrics"][key]["test"]["auc"])
        M[f"lo{m}"], M[f"hi{m}"] = f3(boot["auc"][key]["lo"]), f3(boot["auc"][key]["hi"])
    M["valHeadSelect"] = f3(next(r for r in sel["grids"]["fusion_head"]
                                 if r["max_document_chunks"] == s["fusion_head"]["max_document_chunks"]
                                 and r["head_layers"] == s["fusion_head"]["head_layers"])["val_auc"])
    seeds = [r["test_auc"] for r in test["fusion_head_per_seed"]]
    M["headSeedMin"], M["headSeedMax"] = f3(min(seeds)), f3(max(seeds))
    dh = boot["diff"]["fusion_head - ticker_prior"]
    M.update(diffHeadTick=signed(dh["point"]), diffHeadTickLo=signed(dh["lo"]),
             diffHeadTickHi=signed(dh["hi"]))
    dl = boot["diff"]["fusion_head - logistic_regression"]
    M.update(diffHeadLR=signed(dl["point"]), diffHeadLRLo=signed(dl["lo"]),
             diffHeadLRHi=signed(dl["hi"]))
    for key in ("majority_class", "fusion_head"):
        mt = test["metrics"][key]["test"]
        M[f"brier{MACRO[key]}"], M[f"logloss{MACRO[key]}"] = f3(mt["brier"]), f3(mt["log_loss"])
    M["halfWidthHead"] = f3((boot["auc"]["fusion_head"]["hi"] - boot["auc"]["fusion_head"]["lo"]) / 2)

    # ---- exploratory / controls -----------------------------------------------------
    a = expl["all_with_headlines"]["bootstrap"]
    M.update(explN=a["n_events"], explWeeks=a["n_weeks"],
             explAuc=f3(a["auc"]["zero_shot_headlines"]["point"]),
             explLo=f3(a["auc"]["zero_shot_headlines"]["lo"]),
             explHi=f3(a["auc"]["zero_shot_headlines"]["hi"]))
    w = expl["with_headlines_and_document"]["bootstrap"]
    M.update(explBothN=w["n_events"], explDocAuc=f3(w["auc"]["zero_shot_document"]["point"]),
             explDiff=signed(w["diff"]["zero_shot_headlines - zero_shot_document"]["point"]),
             explDiffLo=signed(w["diff"]["zero_shot_headlines - zero_shot_document"]["lo"]),
             explDiffHi=signed(w["diff"]["zero_shot_headlines - zero_shot_document"]["hi"]))
    neg, pos = ctl["summary"]["synthetic-negative-control"], ctl["summary"]["synthetic-positive-control"]
    M.update(negMean=f3(neg["neural_test_auc"]["mean"]), negSd=f3(neg["neural_test_auc"]["sd"]),
             negMin=f3(neg["neural_test_auc"]["min"]), negMax=f3(neg["neural_test_auc"]["max"]),
             posMean=f3(pos["neural_test_auc"]["mean"]), ctlRuns=neg["n_runs"],
             ctlTestN=int(sum(r["n"]["test"] for r in ctl["runs"]) / len(ctl["runs"])))

    with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
        fh.write("% Generated by experiments/paper_assets.py from results/*.json. Do not edit.\n")
        for k, v in M.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # ---- tables ----------------------------------------------------------------------
    with open(os.path.join(OUT, "tab_split.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lllrr}\n\\toprule\nSplit & First entry & Last entry & $n$ & "
                 "Up-rate \\\\\n\\midrule\n")
        for sp, v in sel["split"]["splits"].items():
            fh.write(f"{sp.capitalize()} & {v['first_entry']} & {v['last_entry']} & {v['n']} & "
                     f"{f3(v['pos_rate'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(OUT, "tab_grid.tex"), "w") as fh:
        fh.write("\\begin{tabular}{llrr}\n\\toprule\nModel & Setting & Chunks & Val AUC \\\\\n"
                 "\\midrule\n")
        for r in sel["grids"]["logistic_regression"]:
            fh.write(f"Logistic regression & $C={r['C']:g}$ & {r['max_document_chunks']} & "
                     f"{f3(r['val_auc'])} \\\\\n")
        fh.write("\\midrule\n")
        for r in sel["grids"]["fusion_head"]:
            fh.write(f"Fusion head & {r['head_layers']} hidden layer{'s' if r['head_layers'] != 1 else ''}"
                     f" & {r['max_document_chunks']} & {f3(r['val_auc'])} $\\pm$ "
                     f"{f3(r['val_auc_sd'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(OUT, "tab_test.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrcrrrrr}\n\\toprule\nModel & Val AUC & Test AUC & 95\\% CI & "
                 "AP & Brier & Log loss & Acc. & F1 \\\\\n\\midrule\n")
        for key, name in MODELS:
            mt, ci = test["metrics"][key]["test"], boot["auc"][key]

            def g(k):
                v = mt.get(k)
                return "--" if v is None else f3(v)
            fh.write(f"{name} & {f3(test['metrics'][key]['val']['auc'])} & {f3(mt['auc'])} & "
                     f"[{f3(ci['lo'])}, {f3(ci['hi'])}] & {g('average_precision')} & {g('brier')} & "
                     f"{g('log_loss')} & {g('accuracy')} & {g('f1')} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    names = dict(MODELS)
    with open(os.path.join(OUT, "tab_diff.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrcr}\n\\toprule\nTest AUC difference & Estimate & 95\\% CI & "
                 "Share $\\le 0$ \\\\\n\\midrule\n")
        for k, dd in boot["diff"].items():
            x, y = k.split(" - ")
            fh.write(f"{names[x]} $-$ {names[y]} & {signed(dd['point'])} & "
                     f"[{signed(dd['lo'])}, {signed(dd['hi'])}] & {dd['frac_le_0']:.2f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(OUT, "tab_controls.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrcccc}\n\\toprule\nControl & Runs & Head test AUC & Range & "
                 "LR test AUC & Head val AUC \\\\\n\\midrule\n")
        for mode, v in ctl["summary"].items():
            n = v["neural_test_auc"]
            fh.write(f"{mode.replace('synthetic-', '').replace('-control', '').capitalize()} & "
                     f"{v['n_runs']} & {f3(n['mean'])} $\\pm$ {f3(n['sd'])} & "
                     f"{f3(n['min'])}--{f3(n['max'])} & "
                     f"{f3(v['logistic_regression_test_auc']['mean'])} & "
                     f"{f3(v['neural_val_auc']['mean'])} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # ---- figures ---------------------------------------------------------------------
    years = d["per_year"]
    ys = list(years)
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    bottom = [0] * len(ys)
    for sp, col in (("train", "#4C72B0"), ("val", "#DD8452"), ("test", "#55A868")):
        vals = [years[y].get(sp, 0) for y in ys]
        ax.bar(ys, vals, bottom=bottom, color=col, label=sp, width=0.7)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("events (by entry year)")
    ax.set_ylim(0, max(bottom) * 1.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_coverage.pdf"), metadata={"CreationDate": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    for i, (key, name) in enumerate(reversed(MODELS)):
        ci = boot["auc"][key]
        ax.plot([ci["lo"], ci["hi"]], [i, i], color="#333333", lw=1.4)
        ax.plot(test["metrics"][key]["test"]["auc"], i, "o", color="#333333", ms=5,
                label="test (95% week-block CI)" if i == 0 else None)
        ax.plot(test["metrics"][key]["val"]["auc"], i, "D", mfc="white", mec="#C44E52", ms=5,
                label="validation" if i == 0 else None)
    ax.axvline(0.5, color="#999999", ls="--", lw=0.8)
    ax.set_yticks(range(len(MODELS)), [n for _, n in reversed(MODELS)])
    ax.set_xlabel("ROC AUC")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_test_auc.pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(OUT, "fig_test_auc.png"), dpi=170, metadata={"Software": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    for j, (mode, lab) in enumerate((("synthetic-negative-control", "negative control"),
                                     ("synthetic-positive-control", "positive control"))):
        runs = [r for r in ctl["runs"] if r["data_mode"] == mode]
        ax.plot([j - 0.08] * len(runs), [r["neural_test_auc"] for r in runs], "o", color="#4C72B0",
                ms=4, label="fusion head" if j == 0 else None)
        ax.plot([j + 0.08] * len(runs), [r["logistic_regression_test_auc"] for r in runs], "s",
                color="#DD8452", ms=4, label="logistic regression" if j == 0 else None)
    ax.axhline(0.5, color="#999999", ls="--", lw=0.8)
    ax.set_xticks([0, 1], ["negative control", "positive control"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("test AUC (per seed)")
    ax.legend(frameon=False, loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_controls.pdf"), metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {len(M)} macros, 5 tables, 3 figures to {OUT}")


if __name__ == "__main__":
    main()
