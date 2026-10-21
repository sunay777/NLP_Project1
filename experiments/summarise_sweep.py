"""Numbers for the write-up, straight from run_collapse JSONs (no hand-copying).

Per real_fraction (final generation, mean +/- sd over seeds): conditional
diversity, off-context mass, marginal diversity, induction (robust + legacy),
causal drop, query accuracy; plus retries, gate failures, circuit-variant
switches, and how often the LEGACY probe would have reported a spurious
'circuit collapse' (legacy < 0.5 while the robust probe >= 0.8).

Also (v2): a late-generation mean (--late, default gens 3-5) beside the final
generation, and PAIRED per-seed differences vs real_fraction = 1. Init seeds are
shared across fractions (common random numbers), so the paired difference
removes the seed effect; it is reported with a sign count (e.g. 3/3 seeds).

Run:  python -m experiments.summarise_sweep --files results/sweep/extended_all*.json
"""
import argparse
import json

import numpy as np

from experiments.plot_collapse import load_many

GET = {
    "cond_div": lambda r: r["cond_diversity"]["cond_entropy"],
    "off_ctx": lambda r: r["cond_diversity"]["off_context"],
    "marg_div": lambda r: r["diversity"]["normalised_entropy"],
    "ind_robust": lambda r: r["circuit"]["induction_best"],
    "ind_legacy": lambda r: r["circuit"]["induction_legacy"],
    "causal": lambda r: r["circuit"]["causal_drop"],
    "causal_L1": lambda r: r["circuit"]["causal_drop_L1"],
    "causal_L1_mean": lambda r: r["circuit"]["causal_drop_L1_mean"],
    "query_nll": lambda r: r["query_nll_all"],
    "test_loss": lambda r: r["test_loss"],
    "acc_all": lambda r: r["query_acc_all"],
    "labels_ok": lambda r: r["pool_label_correct"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--late", type=int, nargs=2, default=[3, 5],
                    help="inclusive generation range for the late-generation mean")
    ap.add_argument("--ref", type=float, default=1.0, help="reference fraction for paired diffs")
    a = ap.parse_args()
    runs = load_many(a.files)
    fracs = sorted({r["real_fraction"] for r in runs})
    out = {"per_fraction": {}, "totals": {}}
    tot = dict(gens=0, retries=0, gate_fail=0, shifted=0, legacy_false_alarm=0)
    for f in fracs:
        rs = [r for r in runs if r["real_fraction"] == f]
        row = {"n_seeds": len(rs)}
        for k, fn in GET.items():
            vals = []
            for r in rs:
                try:
                    vals.append(fn(r["records"][-1]))
                except KeyError:
                    pass
            if vals:
                row[k] = (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4))
            g1 = []
            for r in rs:
                try:
                    g1.append(fn(r["records"][1]))
                except (KeyError, IndexError):
                    pass
            if g1:
                row[k + "_gen1"] = round(float(np.mean(g1)), 4)
        for r in rs:
            for rec in r["records"]:
                tot["gens"] += 1
                tot["retries"] += rec.get("retries", 0)
                tot["gate_fail"] += int(rec.get("converged", True) is False)
                c = rec["circuit"]
                tot["shifted"] += int(c["circuit_type"] == "shifted")
                tot["legacy_false_alarm"] += int(c["induction_legacy"] < 0.5
                                                 and c["induction_best"] >= 0.8)
        out["per_fraction"][f] = row
    out["totals"] = tot

    # late-generation means and paired differences vs the reference fraction
    def late(r, fn):
        v = []
        for rec in r["records"]:
            if a.late[0] <= rec["generation"] <= a.late[1]:
                try:
                    v.append(fn(rec))
                except KeyError:
                    pass
        return float(np.mean(v)) if v else None
    ref = {r["seed"]: r for r in runs if r["real_fraction"] == a.ref}
    out["late"] = {"generations": a.late, "reference_fraction": a.ref, "per_fraction": {}}
    for f in fracs:
        rs = [r for r in runs if r["real_fraction"] == f]
        row = {}
        for k, fn in GET.items():
            vals = {r["seed"]: late(r, fn) for r in rs}
            vals = {s_: v for s_, v in vals.items() if v is not None}
            if not vals:
                continue
            ent = {"mean": round(float(np.mean(list(vals.values()))), 4),
                   "sd": round(float(np.std(list(vals.values()))), 4), "n": len(vals)}
            if f != a.ref:
                d = [vals[s_] - late(ref[s_], fn) for s_ in vals
                     if s_ in ref and late(ref[s_], fn) is not None]
                if d:
                    ent["paired_diff_mean"] = round(float(np.mean(d)), 4)
                    ent["paired_diff_sd"] = round(float(np.std(d)), 4)
                    ent["sign"] = f"{sum(x > 0 for x in d)}+/{sum(x < 0 for x in d)}-/{len(d)}"
            row[k] = ent
        out["late"]["per_fraction"][f] = row
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
