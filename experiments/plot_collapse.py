"""Headline A+C figures from run_collapse output.

  fig_dose_response.png  value at the final generation vs real_fraction (mean +/- 1 sd
                         over seeds): extended-mode conditional diversity and
                         induction strength on ONE shared [0,1] axis, with the
                         base-mode induction curve as the stable control.
                         If both extended curves break at the same fraction, that
                         is the A+C result.
  fig_trajectories.png   per-generation trajectories, one line per real_fraction
                         (light -> dark = more real data), bands = +/- 1 sd.
  summary.csv            the numbers behind both figures (table view).

Both metrics are already normalised to [0,1] (cond. entropy / ln K; attention mass),
so they share a single y-axis — no twin axes.

Run:  python -m experiments.plot_collapse \
          --extended results/collapse_extended_all.json \
          --base results/collapse_base_all.json --out results/figures
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# validated palettes (dataviz reference instance, light mode)
SERIES = {"diversity": "#2a78d6", "induction": "#eb6834", "control": "#1baf7a"}
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # light -> dark
CANON_FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]   # fixed value->colour map (colour follows value)


def frac_color(f, all_fracs):
    ref = CANON_FRACS if set(all_fracs) <= set(CANON_FRACS) else sorted(all_fracs)
    if len(ref) <= len(ORDINAL):
        return ORDINAL[len(ORDINAL) - len(ref) + ref.index(f)]
    return plt.cm.Blues(0.35 + 0.6 * ref.index(f) / (len(ref) - 1))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

METRICS = {
    "cond_diversity": ("Conditional diversity of sq2 (H / ln K)",
                       lambda r: r["cond_diversity"]["cond_entropy"]),
    "slot_entropy": ("Slot entropy of sampled sq2 (/ ln K)",
                     lambda r: r["slot"]["slot_entropy"]),
    "induction_best": ("Induction strength (best L1 head, copy mass)",
                       lambda r: r["circuit"]["induction_best"]),
    "causal_drop": ("Causal drop (acc lost when induction heads ablated)",
                    lambda r: r["circuit"]["causal_drop"]),
    "off_context": ("Off-context mass of sq2 (hallucinated symbols)",
                    lambda r: r["cond_diversity"]["off_context"]),
    "query_acc_all": ("Query accuracy on real data (all K)",
                      lambda r: r["query_acc_all"]),
    "pool_label_correct": ("Fraction of training-pool query labels correct",
                           lambda r: r["pool_label_correct"]),
}


def style():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 6,
    })


def load(path):
    d = json.load(open(path))
    return d["config"], d["runs"]


def load_many(paths):
    runs, seen = [], set()
    for p in paths:
        for r in load(p)[1]:
            k = (r["real_fraction"], r["seed"])
            if k not in seen:                      # first file wins on duplicates
                seen.add(k); runs.append(r)
    return runs


def table(runs, metric):
    """-> fractions (sorted), array (n_frac, n_seed, n_gen), converged mask."""
    fn = METRICS[metric][1]
    fracs = sorted({r["real_fraction"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    G = max(len(r["records"]) for r in runs)
    arr = np.full((len(fracs), len(seeds), G), np.nan)
    conv = np.ones_like(arr, dtype=bool)
    for r in runs:
        i, j = fracs.index(r["real_fraction"]), seeds.index(r["seed"])
        for rec in r["records"]:
            try:
                arr[i, j, rec["generation"]] = fn(rec)
            except KeyError:
                pass
            conv[i, j, rec["generation"]] = rec.get("converged", True) is not False
    return fracs, seeds, arr, conv


def final_stats(arr, last_k=1):
    v = np.nanmean(arr[:, :, -last_k:], axis=2)            # (frac, seed)
    return np.nanmean(v, axis=1), np.nanstd(v, axis=1), v


def _band(ax, x, m, s, color, mk, ls, name):
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, m, ls, marker=mk, color=color, label=name,
            markeredgecolor="white", markeredgewidth=1)


def dose_response(ext_runs, base_runs, out, last_k):
    """(a) diversity + induction on a shared [0,1] axis; (b) off-context mass on
    its own axis (different scale -> separate panel, never a twin axis)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.3),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    specs = [("cond_diversity", ext_runs, "diversity", "o", "-", "Extended: conditional diversity"),
             ("induction_best", ext_runs, "induction", "s", "-", "Extended: induction strength"),
             ("induction_best", base_runs, "control", "^", "--", "Base: induction (control)")]
    rows = []
    fracs = sorted({r["real_fraction"] for r in ext_runs + base_runs})
    X = lambda fs: np.array([fracs.index(f) for f in fs])   # shared x positions
    for metric, runs, key, mk, ls, name in specs:
        if not runs:
            continue
        fr, _, arr, _ = table(runs, metric)
        m, s, per_seed = final_stats(arr, last_k)
        _band(ax, X(fr), m, s, SERIES[key], mk, ls, name)
        for f, mm, ss, ps in zip(fr, m, s, per_seed):
            rows.append([name, f, round(mm, 4), round(ss, 4), len(ps)])
    # gen-0 reference: diversity before any finite-pool / recursive training
    _, _, arr0, _ = table(ext_runs, "cond_diversity")
    ax.axhline(np.nanmean(arr0[:, :, 0]), color=INK2, ls=":", lw=1)
    ax.text(0, np.nanmean(arr0[:, :, 0]) + 0.015, "gen 0 (fresh data)", fontsize=7, color=INK2)

    fracs_o, _, arro, _ = table(ext_runs, "off_context")
    m, s, per_seed = final_stats(arro, last_k)
    _band(bx, X(fracs_o), m, s, SERIES["diversity"], "D", "-",
          "Extended: off-context mass")
    for f, mm, ss, ps in zip(fracs_o, m, s, per_seed):
        rows.append(["Extended: off-context mass", f, round(mm, 4), round(ss, 4), len(ps)])
    bx.axhline(np.nanmean(arro[:, :, 0]), color=INK2, ls=":", lw=1)
    bx.set_ylim(0, max(0.02, float(np.nanmax(m + s)) * 1.25))

    ylab = f"Value at final generation{'' if last_k == 1 else f' (mean of last {last_k})'}"
    for axx in (ax, bx):
        axx.set_xticks(np.arange(len(fracs)))
        axx.set_xticklabels([f"{v:g}" for v in fracs])
        axx.set_xlim(-0.3, len(fracs) - 0.7)
        axx.set_xlabel("Real-data fraction per generation")
    ax.set_ylabel(ylab)
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Diversity vs. induction circuit")
    bx.set_title("(b) Probability mass outside the context")
    bx.set_ylabel("off-context mass")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_dose_response.png"), dpi=200)
    plt.close(fig)
    return rows


def trajectories(panels, out):
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.0),
                             squeeze=False)
    for ax, (runs, metric, title) in zip(axes[0], panels):
        fracs, _, arr, conv = table(runs, metric)
        g = np.arange(arr.shape[2])
        for i, f in enumerate(fracs):
            m, s = np.nanmean(arr[i], 0), np.nanstd(arr[i], 0)
            c = frac_color(f, fracs)
            ax.fill_between(g, m - s, m + s, color=c, alpha=0.12, linewidth=0)
            ax.plot(g, m, marker="o", markersize=4, color=c, label=f"{f:g}")
            bad = ~conv[i].all(0)                          # any seed failed the gate
            if bad.any():
                ax.plot(g[bad], m[bad], "x", color=INK, markersize=6, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_xticks(g)
        if metric != "off_context":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, None)
    for ax, (_, metric, _) in zip(axes[0], panels):
        ax.set_ylabel("mass" if metric == "off_context" else "normalised value")
    axes[0][-1].plot([], [], "x", color=INK, label="gate failed (any seed)")
    axes[0][-1].legend(title="real fraction", fontsize=7, title_fontsize=7,
                       loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_trajectories.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", nargs="+", required=True,
                    help="one or more run_collapse JSONs (runs are concatenated)")
    ap.add_argument("--base", nargs="+", default=None)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--last_k", type=int, default=1,
                    help="average the last k generations for the dose-response")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    ext = load_many(a.extended)
    base = load_many(a.base) if a.base else []

    rows = dose_response(ext, base, a.out, a.last_k)
    panels = [(ext, "cond_diversity", "Extended: conditional diversity"),
              (ext, "induction_best", "Extended: induction strength")]
    panels.append((ext, "off_context", "Extended: off-context mass"))
    if base:
        panels.append((base, "induction_best", "Base: induction (control)"))
    trajectories(panels, a.out)

    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "real_fraction", "mean_final", "sd_final", "n_seeds"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["metric", "real_fraction", "generation", "mean", "sd", "n_not_converged"])
        for runs, tag in [(ext, "extended"), (base, "base")]:
            if not runs:
                continue
            for metric in METRICS:
                try:
                    fracs, _, arr, conv = table(runs, metric)
                except Exception:
                    continue
                if np.isnan(arr).all():
                    continue
                for i, fr in enumerate(fracs):
                    for gg in range(arr.shape[2]):
                        w.writerow([f"{tag}:{metric}", fr, gg,
                                    round(float(np.nanmean(arr[i, :, gg])), 4),
                                    round(float(np.nanstd(arr[i, :, gg])), 4),
                                    int((~conv[i, :, gg]).sum())])
    print(f"figures -> {a.out}/")


if __name__ == "__main__":
    main()
