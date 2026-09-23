"""Numbers for the write-up, straight from run_collapse JSONs (no hand-copying).

Per real_fraction (final generation, mean +/- sd over seeds): conditional
diversity, off-context mass, marginal diversity, induction (robust + legacy),
causal drop, query accuracy; plus retries, gate failures, circuit-variant
switches, and how often the LEGACY probe would have reported a spurious
'circuit collapse' (legacy < 0.5 while the robust probe >= 0.8).

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
    "acc_all": lambda r: r["query_acc_all"],
    "labels_ok": lambda r: r["pool_label_correct"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", default=None)
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
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
