"""Inheritance law: fit  m_{g+1} = a * (1 - lambda) * m_g + eps  to collapse runs.

lambda = real_fraction. Each generation inherits a fraction a of its parent's
error on the synthetic share of its pool and adds eps of its own. The map is a
contraction while a(1 - lambda) < 1, so a critical real fraction exists only if
a > 1, at lambda* = 1 - 1/a.

Per metric (and per group, e.g. temperature) this reports
  * pooled least squares over every (m_g, m_{g+1}) transition: a, eps, R^2, RMSE;
  * per-seed fits;
  * bootstrap 95% CIs for a and eps (trajectories resampled within each
    real_fraction, so every draw keeps all lambda levels);
  * held-out test: fit on --train_fracs, iterate the map from the observed mean
    at the first generation to predict the --test_fracs trajectories; RMSE and
    coverage (prediction within 1 seed-SD / within the bootstrap 95% band);
  * stationary level m* = eps / (1 - a(1-lambda)) and perturbation half-life
    ln 2 / -ln(a(1-lambda)) per lambda.
Transitions touching a generation that fails the strict gate (val accuracy <
0.98 x pool label correctness) are excluded with --exclude_lax and listed.

Run:  python -m experiments.fit_recursion --files results/sweep/extended_all*.json \
          --out results/v2/recursion_fit.json
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np


def _ppl_nll(r):
    return r["query_nll_all"] if "query_nll_all" in r else math.log(r["query_ppl"])


def _syn_wrong(r, lam):
    if "pool_syn_wrong_label" in r:
        return r["pool_syn_wrong_label"]
    if r["generation"] == 0 or lam >= 1:
        return None
    return (1 - r["pool_label_correct"]) / (1 - lam)


# metric -> (description, extractor(record, lambda) -> float | None)
METRICS = {
    "off_context": ("off-context (hallucinated) mass of p(sq2 | context)",
                    lambda r, lam: r.get("cond_diversity", {}).get("off_context")),
    "syn_wrong_label": ("wrong-label rate of the synthetic share of the pool",
                        _syn_wrong),
    "query_nll": ("query NLL on real data (all K queries if logged, else first)",
                  lambda r, lam: _ppl_nll(r)),
    "cond_deficit": ("1 - conditional entropy of sq2 / ln K",
                     lambda r, lam: (1 - r["cond_diversity"]["cond_entropy"])
                     if "cond_diversity" in r else None),
    "pool_offctx": ("off-context rate of sq2 in the synthetic pool (v2)",
                    lambda r, lam: r.get("pool_syn_offctx")),
}


def load_runs(paths):
    runs, seen = [], set()
    for p in paths:
        d = json.load(open(p))
        meta = d.get("config", {})
        for r in d["runs"]:
            key = (meta.get("temperature", 1.0), meta.get("mode"), r["real_fraction"], r["seed"],
                   meta.get("pulse_gen"), meta.get("sq2_zipf", 0.0), os.path.basename(p))
            if key in seen:
                continue
            seen.add(key)
            runs.append({"meta": meta, "file": p, "real_fraction": r["real_fraction"],
                         "seed": r["seed"], "records": r["records"]})
    return runs


def lax(rec, strict=0.98):
    """Fails the strict accuracy gate (val acc < strict x pool label correctness)."""
    if rec["generation"] == 0:
        return rec["attempts"][-1]["val_acc"] < strict if rec.get("attempts") else False
    a = rec["attempts"]
    best = max(a, key=lambda x: x["val_acc"])
    return best["val_acc"] < strict * rec["pool_label_correct"]


def transitions(run, metric, exclude_lax, from_gen=0):
    lam = run["real_fraction"]
    fn = METRICS[metric][1]
    vals = [fn(r, lam) for r in run["records"]]
    out, dropped = [], []
    for g in range(from_gen, len(vals) - 1):
        x, y = vals[g], vals[g + 1]
        if x is None or y is None:
            continue
        if exclude_lax and (lax(run["records"][g]) or lax(run["records"][g + 1])):
            dropped.append((run["real_fraction"], run["seed"], g))
            continue
        out.append((x, y, lam))
    return out, dropped


def fit(pairs):
    if len(pairs) < 3:
        return None
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    lam = np.array([p[2] for p in pairs])
    X = np.stack([x * (1 - lam), np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"a": float(coef[0]), "eps": float(coef[1]),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "rmse": math.sqrt(ss_res / len(y)), "n": int(len(y))}


def iterate(a, eps, lam, m0, n):
    m = [m0]
    for _ in range(n - 1):
        m.append(a * (1 - lam) * m[-1] + eps)
    return m


def stationary(a, eps, lam):
    k = a * (1 - lam)
    return {"gain": k,
            "m_star": eps / (1 - k) if k < 1 else None,
            "half_life": (math.log(2) / -math.log(k)) if 0 < k < 1 else None}


def analyse(runs, metric, args):
    pairs, dropped = [], []
    by_traj = {}
    for r in runs:
        t, d = transitions(r, metric, args.exclude_lax, args.from_gen)
        dropped += d
        if t:
            pairs += t
            by_traj[(r["real_fraction"], r["seed"])] = t
    res = {"metric": metric, "description": METRICS[metric][0],
           "pooled": fit(pairs), "excluded_lax_transitions": dropped}
    if res["pooled"] is None:
        return res

    seeds = sorted({s for _, s in by_traj})
    res["per_seed"] = {str(s): fit([p for (f, ss), t in by_traj.items() if ss == s for p in t])
                       for s in seeds}

    rng = np.random.default_rng(0)
    fracs = sorted({f for f, _ in by_traj})
    groups = {f: [k for k in by_traj if k[0] == f] for f in fracs}
    boot = []
    for _ in range(args.n_boot):
        sample = []
        for f in fracs:
            ks = groups[f]
            for i in rng.integers(0, len(ks), size=len(ks)):
                sample += by_traj[ks[i]]
        b = fit(sample)
        if b:
            boot.append((b["a"], b["eps"]))
    boot = np.array(boot)
    res["bootstrap"] = {"n": int(len(boot)),
                        "a_ci95": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                        "eps_ci95": np.percentile(boot[:, 1], [2.5, 97.5]).tolist()}

    a, eps = res["pooled"]["a"], res["pooled"]["eps"]
    res["critical_lambda"] = (1 - 1 / a) if a > 1 else None
    res["stationary"] = {str(f): stationary(a, eps, f) for f in fracs}

    # held-out fractions: fit on train_fracs only, iterate from the observed gen-start mean
    tr = [p for (f, s), t in by_traj.items() if f in args.train_fracs for p in t]
    ft = fit(tr)
    if ft and any(f in args.test_fracs for f in fracs):
        btr = []
        trk = {f: [k for k in groups.get(f, [])] for f in args.train_fracs}
        for _ in range(args.n_boot):
            sample = []
            for f, ks in trk.items():
                for i in rng.integers(0, len(ks), size=len(ks)) if ks else []:
                    sample += by_traj[ks[i]]
            b = fit(sample)
            if b:
                btr.append((b["a"], b["eps"]))
        held = {"train_fracs": args.train_fracs, "fit": ft, "per_fraction": {}}
        errs, cov1, covb = [], [], []
        fn = METRICS[metric][1]
        for f in args.test_fracs:
            trajs = [r for r in runs if r["real_fraction"] == f]
            if not trajs:
                continue
            series = [[fn(rec, f) for rec in r["records"]] for r in trajs]
            G = min(len(s) for s in series)
            start = next(g for g in range(G) if all(s[g] is not None for s in series))
            obs = np.array([[s[g] for g in range(start, G)] for s in series], dtype=float)
            mu, sd = obs.mean(0), obs.std(0)
            pred = np.array(iterate(ft["a"], ft["eps"], f, mu[0], G - start))
            band = np.array([iterate(a_, e_, f, mu[0], G - start) for a_, e_ in btr])
            lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
            within1 = (np.abs(pred - mu) <= sd)[1:]
            withinb = ((mu >= lo) & (mu <= hi))[1:]
            errs += list((pred - mu)[1:])
            cov1 += list(within1)
            covb += list(withinb)
            held["per_fraction"][str(f)] = {
                "start_generation": start, "observed_mean": mu.tolist(), "observed_sd": sd.tolist(),
                "predicted": pred.tolist(), "band95": [lo.tolist(), hi.tolist()],
                "within_1sd": within1.tolist(), "n_seeds": len(trajs)}
        if errs:
            held["rmse"] = float(np.sqrt(np.mean(np.square(errs))))
            held["coverage_1sd"] = float(np.mean(cov1))
            held["coverage_band95"] = float(np.mean(covb))
            res["heldout"] = held
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--metrics", nargs="+", default=list(METRICS))
    ap.add_argument("--group_by", default=None,
                    help="config key to fit separately per value (e.g. temperature)")
    ap.add_argument("--train_fracs", type=float, nargs="+", default=[0.0, 0.25, 1.0])
    ap.add_argument("--test_fracs", type=float, nargs="+", default=[0.1, 0.5])
    ap.add_argument("--from_gen", type=int, default=0,
                    help="ignore transitions starting before this generation")
    ap.add_argument("--exclude_lax", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted({f for p in args.files for f in glob.glob(p)})
    runs = load_runs(files)
    groups = {"all": runs}
    if args.group_by:
        groups = {}
        for r in runs:
            groups.setdefault(str(r["meta"].get(args.group_by)), []).append(r)
    out = {"files": files, "args": vars(args), "groups": {}}
    rows = []
    for gk, rs in sorted(groups.items()):
        out["groups"][gk] = {}
        for m in args.metrics:
            res = analyse(rs, m, args)
            out["groups"][gk][m] = res
            p = res.get("pooled")
            if not p:
                continue
            h = res.get("heldout", {})
            ci = res.get("bootstrap", {})
            rows.append([gk, m, round(p["a"], 4), round(p["eps"], 5), round(p["r2"], 4),
                         round(p["rmse"], 5), p["n"],
                         *[round(v, 4) for v in ci.get("a_ci95", [float("nan")] * 2)],
                         *[round(v, 5) for v in ci.get("eps_ci95", [float("nan")] * 2)],
                         round(h.get("rmse", float("nan")), 5),
                         round(h.get("coverage_1sd", float("nan")), 3),
                         round(h.get("coverage_band95", float("nan")), 3),
                         len(res["excluded_lax_transitions"])])
            print(f"[{gk}] {m:16s} a={p['a']:.3f} eps={p['eps']:.4f} R2={p['r2']:.3f} "
                  f"RMSE={p['rmse']:.4f} n={p['n']} | a95={ci.get('a_ci95')} "
                  f"| heldout RMSE={h.get('rmse')} cov1sd={h.get('coverage_1sd')}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    with open(os.path.splitext(args.out)[0] + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "metric", "a", "eps", "r2", "rmse", "n", "a_lo", "a_hi",
                    "eps_lo", "eps_hi", "heldout_rmse", "heldout_cov_1sd",
                    "heldout_cov_band95", "n_excluded_lax"])
        w.writerows(rows)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
