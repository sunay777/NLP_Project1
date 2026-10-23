"""Numbers + figures for every v2 experiment, straight from results/v2 (no hand-copying).

  python -m experiments.analyse_v2 all            # everything that has data
  python -m experiments.analyse_v2 temperature    # scores the PRE-REGISTERED tests
  (others: main pulse scrub accumulate zipf poolsize synthctx scale)

Writes results/v2/analysis/<experiment>.json and results/v2/figures/fig_<experiment>.png.
Paired comparisons use common seeds: runs with the same seed share gen 0 and every
pool up to the point where the conditions differ (CPU, 1 thread, deterministic).
"""
import argparse
import glob
import json
import math
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.fit_recursion import load_runs, analyse, iterate, METRICS
from experiments.plot_headline import style, colour, MARKERS, INK2

R = "results/v2"
A = f"{R}/analysis"
FIG = f"{R}/figures"


def runs_of(name):
    return load_runs(sorted(glob.glob(f"{R}/{name}/*.json")))


def fitargs(**kw):
    d = dict(train_fracs=[0.0, 0.25, 1.0], test_fracs=[0.1, 0.5], from_gen=0,
             exclude_lax=False, n_boot=1000)
    d.update(kw)
    return SimpleNamespace(**d)


def metric_series(runs, fn, f=None, G=None):
    rs = [r for r in runs if f is None or r["real_fraction"] == f]
    if not rs:
        return None, []
    G = G or min(len(r["records"]) for r in rs)
    arr = np.array([[fn(rec) if fn(rec) is not None else np.nan for rec in r["records"][:G]]
                    for r in rs], dtype=float)
    return arr, [r["seed"] for r in rs]


OFF = lambda r: r["cond_diversity"]["off_context"]
WRONG = lambda r: r.get("pool_syn_wrong_label")
CONDH = lambda r: r["cond_diversity"]["cond_entropy"]


def save(name, obj):
    os.makedirs(A, exist_ok=True)
    json.dump(obj, open(f"{A}/{name}.json", "w"), indent=1, default=float)
    print(f"-> {A}/{name}.json")


def savefig(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_{name}.png", dpi=300)
    fig.savefig(f"{FIG}/fig_{name}.pdf")
    plt.close(fig)
    print(f"-> {FIG}/fig_{name}.png")


def law_fit(runs, metric="off_context", **kw):
    return analyse(runs, metric, fitargs(**kw))


# ----------------------------------------------------------------------------
def do_main():
    out = {}
    for mode in ("main_ext", "main_base"):
        rs = runs_of(mode)
        if not rs:
            continue
        ms = ["off_context", "syn_wrong_label", "query_nll", "cond_deficit", "pool_offctx"] \
            if mode == "main_ext" else ["syn_wrong_label", "query_nll"]
        out[mode] = {m: law_fit(rs, m) for m in ms}
        # plateau test (lambda = 0, 20 generations): observed late mean vs predicted m*
        if mode == "main_ext":
            f = out[mode]["off_context"]
            p = f["pooled"]
            arr, _ = metric_series(rs, OFF, 0.0)
            if arr is not None and arr.shape[1] >= 16:
                out[mode]["plateau_lambda0"] = {
                    "predicted_m_star": f["stationary"]["0.0"]["m_star"],
                    "observed_mean_gens_16_19": float(np.nanmean(arr[:, 16:20])),
                    "observed_sd_seed_means": float(np.nanstd(np.nanmean(arr[:, 16:20], 1))),
                    "late_slope_per_gen": float(np.polyfit(np.arange(12, arr.shape[1]),
                                                           np.nanmean(arr[:, 12:], 0), 1)[0]),
                    "early_slope_per_gen": float(np.polyfit(np.arange(1, 6),
                                                            np.nanmean(arr[:, 1:6], 0), 1)[0])}
        gate = [rec for r in rs for rec in r["records"]]
        out[mode]["gate"] = {"generations": len(gate),
                             "retries": int(sum(r.get("retries", 0) for r in gate)),
                             "not_converged": int(sum(r.get("converged") is False for r in gate))}
    save("main", out)


# ----------------------------------------------------------------------------
def do_temperature():
    pred_p = "paper/PREDICTION.json"
    if not os.path.exists(pred_p):
        print("no paper/PREDICTION.json -- nothing to score")
        return
    pred = json.load(open(pred_p))
    main = runs_of("main_ext")
    byT = {1.0: [dict(r, records=r["records"][:10]) for r in main if r["seed"] in range(5)
                 and r["real_fraction"] in (0.0, 0.25, 0.5)]}
    for T in (0.7, 1.3, 1.6):
        byT[T] = runs_of(f"temp_T{T:g}")
    byT = {T: rs for T, rs in byT.items() if rs}
    out = {"fits": {}, "tests": {}}
    from experiments.fit_recursion import transitions, fit
    mx = pred.get("max_m_fit", 1.0)
    for T, rs in byT.items():           # same linear-regime rule as the pre-registration
        pairs = [p for r in rs for p in transitions(r, "off_context", False)[0] if p[0] < mx]
        f = fit(pairs)
        out["fits"][f"{T:g}"] = {"pooled": f, "critical_lambda": (1 - 1 / f["a"]) if f["a"] > 1 else None}
    Ts = sorted(byT)
    a_conf = [out["fits"][f"{T:g}"]["pooled"]["a"] for T in Ts]
    out["tests"]["1_gain_monotone_in_T"] = {"T": Ts, "a": a_conf,
                                             "pass": bool(len(Ts) >= 2 and all(np.diff(a_conf) > 0))}
    # test 2: inside the predicted band at >= 80% of gens 1-9, in >= 7/9 cells
    cells, cell_pass = {}, []
    for T in Ts:
        P = pred["temperatures"].get(f"{T:g}")
        if not P:
            continue
        for lam in pred["confirmatory"]["lambdas"]:
            arr, _ = metric_series(byT[T], OFF, lam, 10)
            if arr is None:
                continue
            mu = np.nanmean(arr, 0)
            lo, hi = map(np.array, P["trajectories"][f"{lam:g}"]["band95"])
            pt = np.array(P["trajectories"][f"{lam:g}"]["point"])
            use = (pt <= pred.get("max_m_test", 1.0))[1:10]
            inside = ((mu >= lo) & (mu <= hi))[1:10][use]
            ok = bool(inside.size and inside.mean() >= 0.8)
            cells[f"T{T:g}_l{lam:g}"] = {"observed": mu.tolist(), "band": [lo.tolist(), hi.tolist()],
                                         "inside_frac": float(inside.mean()), "pass": ok}
            cell_pass.append(ok)
    out["tests"]["2_trajectories_in_band"] = {"cells": cells, "n_pass": int(sum(cell_pass)),
                                              "n_cells": len(cell_pass),
                                              "pass": bool(sum(cell_pass) >= 7)}
    # test 3: contraction vs supercritical regime by slopes
    t3 = {}
    for T in Ts:
        P = pred["temperatures"].get(f"{T:g}")
        if not P:
            continue
        for lam in pred["confirmatory"]["lambdas"]:
            arr, seeds = metric_series(byT[T], OFF, lam, 10)
            if arr is None:
                continue
            gain = P["a"] * (1 - lam)
            if gain >= 1:
                continue                          # supercritical cells are scored by test 4
            early = [np.polyfit(range(1, 5), a_[1:5], 1)[0] for a_ in arr]
            late = [np.polyfit(range(6, 10), a_[6:10], 1)[0] for a_ in arr]
            smaller = sum(l < e for e, l in zip(early, late))
            ok = smaller >= 4
            t3[f"T{T:g}_l{lam:g}"] = {"predicted_gain": gain, "late_smaller_in": f"{smaller}/{len(arr)}",
                                      "pass": bool(ok)}
    out["tests"]["3_regime"] = {"cells": t3, "pass": bool(t3 and all(v["pass"] for v in t3.values()))}
    # test 4: runaway (supercritical) vs bounded (contracting), incl. the lambda* +- 0.1 cells
    t4 = {}
    for T in Ts:
        P = pred["temperatures"].get(f"{T:g}")
        if not P:
            continue
        for lam in sorted({r["real_fraction"] for r in byT[T]}):
            arr, _ = metric_series(byT[T], OFF, lam, 10)
            if arr is None or arr.shape[1] < 10:
                continue
            gain = P["a"] * (1 - lam)
            if gain >= 1:
                n = int((arr[:, 9] > pred.get("max_m_test", 0.5)).sum())
                kind = "supercritical"
            else:
                if f"{lam:g}" not in P["trajectories"]:
                    continue                      # no pre-registered band for this cell
                bound = P["trajectories"][f"{lam:g}"]["band95"][1][9]
                n = int((arr[:, 9] <= bound).sum())
                kind = "contracting"
            t4[f"T{T:g}_l{lam:g}"] = {"predicted_gain": gain, "kind": kind,
                                      "seeds_consistent": f"{n}/{len(arr)}", "pass": bool(n >= 4)}
    out["tests"]["4_runaway_vs_bounded"] = {"cells": t4,
                                            "pass": bool(t4 and all(v["pass"] for v in t4.values()))}
    # test 4: hallucination vs homogenisation at gen 9, lambda = 0
    offs, hs = [], []
    for T in Ts:
        a_off, _ = metric_series(byT[T], OFF, 0.0, 10)
        a_h, _ = metric_series(byT[T], CONDH, 0.0, 10)
        offs.append(float(np.nanmean(a_off[:, 9])))
        hs.append(float(np.nanmean(a_h[:, 9])))
    out["tests"]["5_tradeoff"] = {"T": Ts, "off_context_gen9": offs, "cond_entropy_gen9": hs,
                                  "pass": bool(all(np.diff(offs) > 0) and all(np.diff(hs) > 0))}
    save("temperature", out)

    # phase diagram: gain a(T)(1 - lambda) with the predicted boundary
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    lam_grid = np.linspace(0, 1, 101)
    for i, T in enumerate(Ts):
        a_ = out["fits"][f"{T:g}"]["pooled"]["a"]
        ax[0].plot(lam_grid, a_ * (1 - lam_grid), color=plt.cm.viridis(i / max(1, len(Ts) - 1)),
                   label=f"T={T:g}: a={a_:.2f}")
    ax[0].axhline(1, color=INK2, ls="--", lw=0.8)
    ax[0].text(0.98, 1.02, "critical: a(1-λ)=1", ha="right", va="bottom", fontsize=7, color=INK2,
               transform=ax[0].get_yaxis_transform())
    ax[0].set_xlabel("real-data fraction λ")
    ax[0].set_ylabel("inheritance gain a(T)(1-λ)")
    ax[0].set_title("(a) Gain per (T, λ)")
    ax[0].legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    for i, T in enumerate(Ts):
        c = plt.cm.viridis(i / max(1, len(Ts) - 1))
        ax[1].scatter(offs[i], hs[i], color=c, s=18, zorder=3)
        # T=1.3 and T=1.6 land close together: put every other label below-left
        below = i % 2 == 0 and i > 0
        ax[1].annotate(f"T={T:g}", (offs[i], hs[i]), textcoords="offset points",
                       xytext=(-4, -9) if below else (4, 2), ha="right" if below else "left",
                       fontsize=7)
    ax[1].set_xlabel("off-context mass (gen 9, λ=0)")
    ax[1].set_ylabel("conditional entropy of sq2")
    ax[1].set_title("(b) Hallucination vs homogenisation")
    savefig(fig, "temperature")


# ----------------------------------------------------------------------------
def paired(cond, ctrl, fn, G):
    """seed-matched difference cond - ctrl per generation -> {lam: (n_seeds, G)}"""
    out = {}
    for lam in sorted({r["real_fraction"] for r in cond}):
        c = {r["seed"]: r for r in cond if r["real_fraction"] == lam}
        k = {r["seed"]: r for r in ctrl if r["real_fraction"] == lam}
        ss = sorted(set(c) & set(k))
        if not ss:
            continue
        d = np.array([[(fn(c[s]["records"][g]) or 0) - (fn(k[s]["records"][g]) or 0)
                       for g in range(G)] for s in ss])
        out[lam] = (ss, d)
    return out


def decay_fit(excess, start=1):
    """geometric decay ratio of a positive excess from generation `start` on."""
    m = np.nanmean(excess, 0)[start:]
    g = np.arange(len(m))
    ok = m > 0
    if ok.sum() < 3:
        return None
    slope = np.polyfit(g[ok], np.log(m[ok]), 1)[0]
    return float(math.exp(slope))


def do_pulse():
    a_main = json.load(open(f"{A}/main.json"))["main_ext"]["off_context"]["pooled"]["a"] \
        if os.path.exists(f"{A}/main.json") else None
    out = {"a_from_main_sweep": a_main}
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for k, (cond, ctrl, fn, label) in enumerate([
            ("pulse_ext", "main_ext", WRONG, "extended: wrong-label rate"),
            ("pulse_ext", "main_ext", OFF, "extended: off-context mass"),
            ("pulse_base", "main_base", WRONG, "base: wrong-label rate")]):
        c, m = runs_of(cond), runs_of(ctrl)
        if not c or not m:
            continue
        P = paired(c, m, fn, 8)
        res = {}
        fr = sorted(P)
        for i, lam in enumerate(fr):
            ss, d = P[lam]
            r_obs = decay_fit(d, start=1)
            r_pred = (1 - lam) * a_main if a_main else None
            res[f"{lam:g}"] = {"seeds": ss, "excess_mean": np.nanmean(d, 0).tolist(),
                               "excess_sd": np.nanstd(d, 0).tolist(),
                               "decay_ratio_observed": r_obs, "decay_ratio_predicted": r_pred,
                               "half_life_observed": (math.log(2) / -math.log(r_obs))
                               if r_obs and 0 < r_obs < 1 else None,
                               "half_life_predicted": (math.log(2) / -math.log(r_pred))
                               if r_pred and 0 < r_pred < 1 else None}
            mu, sd = np.nanmean(d, 0), np.nanstd(d, 0)
            g = np.arange(len(mu))
            col = colour(lam, [0.0, 0.25, 0.5])
            ax[k].errorbar(g, mu, yerr=sd, fmt=MARKERS[i], color=col, ms=3, lw=0.8, ls="-",
                           elinewidth=0.6, label=f"λ={lam:g}")
            if r_pred and mu[1] > 0:
                ax[k].plot(g[1:], mu[1] * r_pred ** (g[1:] - 1), "--", color=col, lw=0.8)
        out[f"{cond}:{label}"] = res
        ax[k].set_title(label, fontsize=7)
        ax[k].set_xlabel("generation")
        ax[k].set_ylabel("pulsed − unpulsed (paired)")
    ax[0].legend(fontsize=6.5)
    save("pulse", out)
    savefig(fig, "pulse")


def do_scrub():
    main = runs_of("main_ext")
    out = {}
    for sc in ("offctx", "labels"):
        rs = runs_of(f"scrub_{sc}")
        if not rs:
            continue
        P = paired(rs, main, OFF, 12)
        ss, d = P.get(0.0, ([], None))
        out[sc] = {"law_fit_scrubbed": law_fit(rs, train_fracs=[0.0], test_fracs=[])["pooled"],
                   "paired_offctx_diff_vs_main": None if d is None else
                   {"seeds": ss, "mean": np.nanmean(d, 0).tolist(), "sd": np.nanstd(d, 0).tolist()}}
        arr, _ = metric_series(rs, OFF, 0.0, 12)
        out[sc]["offctx_mean"] = np.nanmean(arr, 0).tolist()
    save("scrub", out)


def do_accumulate():
    rs, main = runs_of("accumulate"), runs_of("main_ext")
    if not rs:
        return
    out = {"law_fit_accumulate": law_fit(rs, train_fracs=[0.0, 0.25], test_fracs=[])["pooled"]}
    for lam, (ss, d) in paired(rs, main, OFF, 12).items():
        out[f"paired_offctx_diff_l{lam:g}"] = {"seeds": ss, "mean": np.nanmean(d, 0).tolist(),
                                               "sd": np.nanstd(d, 0).tolist()}
    save("accumulate", out)


def do_zipf():
    out = {}
    for al in ("1", "2"):
        rs = runs_of(f"zipf{al}")
        if not rs:
            continue
        o = {"law_fit": law_fit(rs, train_fracs=[0.0, 1.0], test_fracs=[])["pooled"]}
        for lam in (0.0, 1.0):
            for k, fn in (("tail_ratio", lambda r: r["cond_diversity"]["tail_ratio"]),
                          ("head_ratio", lambda r: r["cond_diversity"]["head_ratio"]),
                          ("kl_true", lambda r: r["cond_diversity"]["kl_true"]),
                          ("off_context", OFF)):
                arr, ss = metric_series(rs, fn, lam, 10)
                if arr is not None:
                    o[f"{k}_l{lam:g}"] = {"mean": np.nanmean(arr, 0).tolist(),
                                          "sd": np.nanstd(arr, 0).tolist(), "seeds": ss}
        out[f"alpha{al}"] = o
    save("zipf", out)


def do_poolsize():
    out = {}
    for name in ("pool5k", "main_ext", "pool80k"):
        rs = [r for r in runs_of(name) if r["real_fraction"] in (0.0, 1.0)]
        if name == "main_ext":
            rs = [dict(r, records=r["records"][:10]) for r in rs if r["seed"] < 5]
        if rs:
            out[name] = law_fit(rs, train_fracs=[0.0, 1.0], test_fracs=[])["pooled"]
    save("poolsize", out)


def do_synthctx():
    out = {}
    for name in ("fulllm_promptreal", "fulllm_synthctx"):
        rs = runs_of(name)
        if not rs:
            continue
        o = {}
        for lam in sorted({r["real_fraction"] for r in rs}):
            for k, fn in (("acc_all", lambda r: r["query_acc_all"]),
                          ("causal_L1_mean", lambda r: r["circuit"]["causal_drop_L1_mean"]),
                          ("selfgen_wellformed", lambda r: r.get("selfgen_wellformed")),
                          ("pool_syn_wellformed", lambda r: r.get("pool_syn_wellformed")),
                          ("pool_label_correct", lambda r: r.get("pool_label_correct")),
                          ("off_context", OFF), ("query_nll", lambda r: r["query_nll_all"])):
                arr, ss = metric_series(rs, fn, lam, 10)
                if arr is not None:
                    o[f"{k}_l{lam:g}"] = {"mean": np.nanmean(arr, 0).tolist(),
                                          "sd": np.nanstd(arr, 0).tolist(), "seeds": ss}
            o["not_converged"] = int(sum(rec.get("converged") is False
                                         for r in rs for rec in r["records"]))
        out[name] = o
    save("synthctx", out)


def do_scale():
    out = {}
    for name in sorted({os.path.basename(p) for p in glob.glob(f"{R}/arch_*") + glob.glob(f"{R}/task_*")}):
        rs = runs_of(name)
        if not rs:
            continue
        o = {"n_runs": len(rs),
             "not_converged": int(sum(rec.get("converged") is False for r in rs for rec in r["records"]))}
        if len({r["real_fraction"] for r in rs}) > 1:
            o["law_fit"] = law_fit(rs, train_fracs=[0.0, 1.0], test_fracs=[])["pooled"]
        o["gen0_acc"] = float(np.mean([r["records"][0]["query_acc_all"] for r in rs]))
        out[name] = o
    save("scale", out)


TASKS = {"main": do_main, "temperature": do_temperature, "pulse": do_pulse, "scrub": do_scrub,
         "accumulate": do_accumulate, "zipf": do_zipf, "poolsize": do_poolsize,
         "synthctx": do_synthctx, "scale": do_scale}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=list(TASKS) + ["all"])
    a = ap.parse_args()
    style()
    for k in (TASKS if a.what == "all" else [a.what]):
        try:
            TASKS[k]()
        except Exception as e:                       # one missing experiment must not stop the rest
            print(f"[{k}] skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
