"""Composite headline figure for the 2-page abstract (7.0 x 2.5 in, fonts >= 7 pt).

  (a) off-context mass per generation, with the inheritance law
      m_{g+1} = a(1-lambda) m_g + eps fitted on the TRAIN fractions only:
      solid = fitted fractions, dashed = held-out predictions (recursion_fit.json);
  (b) query NLL on real data (log y): calibration erodes while accuracy stays ~1;
  (c) variant-agnostic circuit dependence: accuracy lost when ALL layer-1 heads
      are ablated (mean-ablation when logged, else zero-ablation; for the
      original sweep only the re-probed gen-0 / final models have it);
  (d) conditional diversity of sq2 on a y-range that shows the signal.
Real fractions use one sequential hue (darkest at lambda=0, the most collapsed) plus a
distinct marker per fraction, so lines stay identifiable in greyscale and for
colour-blind readers.

Run:  python -m experiments.plot_headline --runs "results/sweep/extended_all*.json" \
          --fit results/v2/recursion_fit.json \
          --reprobe results/figures/reprobe_extended.csv --out results/v2/figures/fig_headline.png
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.fit_recursion import load_runs, iterate

BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style():
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 1.0, "lines.markersize": 3.2, "axes.titlelocation": "left",
    })


def colour(f, fracs, dark_first=False):
    ref = sorted(fracs, reverse=dark_first)
    if len(ref) <= len(BLUES):
        return BLUES[len(BLUES) - len(ref) + ref.index(f)]
    return plt.cm.Blues(0.35 + 0.6 * ref.index(f) / (len(ref) - 1))


def series(runs, f, fn):
    rs = [r for r in runs if r["real_fraction"] == f]
    G = max(len(r["records"]) for r in rs)
    a = np.full((len(rs), G), np.nan)
    for i, r in enumerate(rs):
        for rec in r["records"]:
            try:
                v = fn(rec)
            except (KeyError, TypeError):
                v = None
            if v is not None:
                a[i, rec["generation"]] = v
    return a


def points(ax, g, a, c, mk, label=None, line=True):
    m, s = np.nanmean(a, 0), np.nanstd(a, 0)
    ok = ~np.isnan(m)
    ax.errorbar(g[ok], m[ok], yerr=s[ok], fmt=mk, color=c, ms=3.2, elinewidth=0.6,
                capsize=0, markeredgecolor="white", markeredgewidth=0.4, label=label,
                ls="-" if line else "none", lw=0.8 if line else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--reprobe", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    style()
    runs = load_runs(sorted({f for p in a.runs for f in glob.glob(p)}))
    fracs = sorted({r["real_fraction"] for r in runs})
    fit = json.load(open(a.fit))["groups"]["all"]["off_context"]
    held = fit.get("heldout", {})
    ftr = held.get("fit", fit["pooled"])
    train = set(held.get("train_fracs", fracs))

    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.45))
    # (a) off-context mass + law
    for i, f in enumerate(fracs):
        c, mk = colour(f, fracs, dark_first=True), MARKERS[i]
        arr = series(runs, f, lambda r: r["cond_diversity"]["off_context"])
        g = np.arange(arr.shape[1])
        points(ax[0], g, arr, c, mk, label=f"{f:g}", line=False)
        mu = np.nanmean(arr, 0)
        pred = iterate(ftr["a"], ftr["eps"], f, mu[0], len(g))
        ax[0].plot(g, pred, "-" if f in train else "--", color=c, lw=1.0)
    ax[0].set_title("(a) Off-context mass")
    ax[0].set_ylabel("mass outside the context")
    ax[0].text(0.03, 0.97, f"a={ftr['a']:.2f}\n$\\varepsilon$={ftr['eps']:.3f}",
               transform=ax[0].transAxes, va="top", fontsize=7, color=INK2)

    # (b) query NLL, log y
    for i, f in enumerate(fracs):
        arr = series(runs, f, lambda r: r.get("query_nll_all", np.log(r["query_ppl"])))
        points(ax[1], np.arange(arr.shape[1]), arr, colour(f, fracs, dark_first=True), MARKERS[i])
    ax[1].set_yscale("log")
    ax[1].set_ylabel("NLL (log scale)")
    accs = [rec["query_acc_all"] for r in runs for rec in r["records"]]
    ax[1].set_title(f"(b) Query NLL, acc$\\geq${min(accs):.2f}")

    # (c) variant-agnostic causal dependence on layer 1
    key = None
    for k in ("causal_drop_L1_mean", "causal_drop_L1"):
        if all(k in rec["circuit"] for r in runs for rec in r["records"]):
            key = k
            break
    if key:
        for i, f in enumerate(fracs):
            arr = series(runs, f, lambda r: r["circuit"][key])
            points(ax[2], np.arange(arr.shape[1]), arr, colour(f, fracs, dark_first=True), MARKERS[i])
        ax[2].set_title("(c) Ablate all L1 heads")
    elif a.reprobe:
        rows = list(csv.DictReader(open(a.reprobe)))
        gens = sorted({int(r["generation"]) for r in rows})
        for i, f in enumerate(fracs):
            vals = [[float(r["causal_drop_L1"]) for r in rows
                     if float(r["real_fraction"]) == f and int(r["generation"]) == g] for g in gens]
            x = np.array(gens) + (i - len(fracs) / 2) * 0.12
            m = [np.mean(v) for v in vals]
            s = [np.std(v) for v in vals]
            ax[2].errorbar(x, m, yerr=s, fmt=MARKERS[i], color=colour(f, fracs, dark_first=True), ms=3.2,
                           elinewidth=0.6, markeredgecolor="white", markeredgewidth=0.4)
        ax[2].set_xticks(gens)
        ax[2].set_title("(c) Ablate all L1 heads")
    ax[2].set_ylim(0, 1.0)
    ax[2].set_ylabel("accuracy lost" + (" (mean-abl.)" if key and "mean" in key else " (zero-abl.)"))

    # (d) conditional diversity, tight y-range
    lo = 1.0
    for i, f in enumerate(fracs):
        arr = series(runs, f, lambda r: r["cond_diversity"]["cond_entropy"])
        points(ax[3], np.arange(arr.shape[1]), arr, colour(f, fracs, dark_first=True), MARKERS[i])
        lo = min(lo, float(np.nanmin(np.nanmean(arr, 0) - np.nanstd(arr, 0))))
    ax[3].set_ylim(max(0.0, lo - 0.02), 1.005)
    ax[3].set_title("(d) Cond. diversity")
    ax[3].set_ylabel("H(sq2 | ctx) / ln K")

    for x in ax:
        x.set_xlabel("generation")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=colour(f, fracs, dark_first=True), marker=MARKERS[i], ms=3.5, lw=1.0,
                markeredgecolor="white", markeredgewidth=0.4) for i, f in enumerate(fracs)]
    h += [Line2D([], [], color=INK2, lw=1.0), Line2D([], [], color=INK2, lw=1.0, ls="--")]
    l = [f"{f:g}" for f in fracs] + ["law (fitted)", "law (held-out)"]
    fig.legend(h, l, title="real-data fraction $\\lambda$", ncol=len(fracs) + 2, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), fontsize=7, title_fontsize=7, handletextpad=0.2,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=0.4)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(os.path.splitext(a.out)[0] + ".pdf")
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
