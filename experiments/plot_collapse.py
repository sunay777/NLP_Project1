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


def dose_response(ext_runs, base_runs, out, last_k):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    specs = [("cond_diversity", ext_runs, "diversity", "o", "-", "Extended: diversity"),
             ("induction_best", ext_runs, "induction", "s", "-", "Extended: induction"),
             ("induction_best", base_runs, "control", "^", "--", "Base: induction (control)")]
    rows = []
    for metric, runs, key, mk, ls, name in specs:
        if not runs:
            continue
        fracs, _, arr, _ = table(runs, metric)
        m, s, per_seed = final_stats(arr, last_k)
        x = np.arange(len(fracs))
        ax.fill_between(x, m - s, m + s, color=SERIES[key], alpha=0.15, linewidth=0)
        ax.plot(x, m, ls, marker=mk, color=SERIES[key], label=name,
                markeredgecolor="white", markeredgewidth=1)
        for f, mm, ss, ps in zip(fracs, m, s, per_seed):
            rows.append([name, f, round(mm, 4), round(ss, 4), len(ps)])
    ax.set_xticks(np.arange(len(fracs)))
    ax.set_xticklabels([f"{f:g}" for f in fracs])
    ax.set_xlabel("Real-data fraction per generation (evenly spaced)")
    ax.set_ylabel(f"Value at final generation{'' if last_k == 1 else f' (mean of last {last_k})'}")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.3, len(fracs) - 0.7)
    ax.set_title("Collapse dose-response: diversity vs. induction circuit")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_dose_response.png"), dpi=200)
    plt.close(fig)
    return rows


def trajectories(panels, out):
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.0),
                             squeeze=False, sharey=True)
    for ax, (runs, metric, title) in zip(axes[0], panels):
        fracs, _, arr, conv = table(runs, metric)
        cols = ORDINAL[-len(fracs):] if len(fracs) <= len(ORDINAL) else \
            plt.cm.Blues(np.linspace(0.35, 0.95, len(fracs)))
        g = np.arange(arr.shape[2])
        for i, f in enumerate(fracs):
            m, s = np.nanmean(arr[i], 0), np.nanstd(arr[i], 0)
            ax.fill_between(g, m - s, m + s, color=cols[i], alpha=0.12, linewidth=0)
            ax.plot(g, m, marker="o", markersize=4, color=cols[i], label=f"{f:g}")
            bad = ~conv[i].all(0)                          # any seed failed the gate
            if bad.any():
                ax.plot(g[bad], m[bad], "x", color=INK, markersize=6, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_xticks(g)
        ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("Normalised value")
    axes[0][-1].plot([], [], "x", color=INK, label="gate failed (any seed)")
    axes[0][-1].legend(title="real fraction", fontsize=7, title_fontsize=7,
                       loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_trajectories.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--last_k", type=int, default=1,
                    help="average the last k generations for the dose-response")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    _, ext = load(a.extended)
    base = load(a.base)[1] if a.base else []

    rows = dose_response(ext, base, a.out, a.last_k)
    panels = [(ext, "cond_diversity", "Extended: conditional diversity"),
              (ext, "induction_best", "Extended: induction strength")]
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
