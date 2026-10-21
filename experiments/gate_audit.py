"""Audit the convergence gate of an existing sweep against the strict v2 gate.

The sweep's gate (0.9 x pool label correctness) passed some models that had not
fully learned the task. The strict v2 gate requires
  (1) validation accuracy >= 0.98 x pool label correctness (gen 0: >= 0.98), and
  (2) causal_drop_L1 >= 0.5 on real data (only checkable where it was logged:
      v2 records, or the re-probed gen-0 / final models in --reprobe CSVs).
For every model that fails, the CSV lists the downstream effect: the synthetic
wrong-label rate of the pool it generated for the next generation, against the
median for the same (mode, real_fraction).

Run:  python -m experiments.gate_audit --files results/sweep/base_all.json \
          "results/sweep/extended_all*.json" \
          --reprobe results/figures/reprobe_base.csv results/figures/reprobe_extended.csv \
          --out results/v2/gate_audit.csv
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def syn_wrong(rec, lam):
    if "pool_syn_wrong_label" in rec:
        return rec["pool_syn_wrong_label"]
    if rec["generation"] == 0 or lam >= 1:
        return None
    return (1 - rec["pool_label_correct"]) / (1 - lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--reprobe", nargs="*", default=[])
    ap.add_argument("--strict", type=float, default=0.98)
    ap.add_argument("--causal", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    causal = {}
    for p in a.reprobe:
        mode = "base" if "base" in os.path.basename(p) else "extended"
        for r in csv.DictReader(open(p)):
            causal[(mode, float(r["real_fraction"]), int(r["seed"]), int(r["generation"]))] = \
                float(r["causal_drop_L1"])

    files = sorted({f for p in a.files for f in glob.glob(p)})
    rows, n_models = [], defaultdict(int)
    med = defaultdict(list)
    runs = []
    for p in files:
        d = json.load(open(p))
        mode = d["config"]["mode"]
        for r in d["runs"]:
            runs.append((mode, p, r))
            for rec in r["records"]:
                w = syn_wrong(rec, r["real_fraction"])
                if w is not None:
                    med[(mode, r["real_fraction"])].append(w)
    for mode, p, r in runs:
        lam, seed, recs = r["real_fraction"], r["seed"], r["records"]
        for i, rec in enumerate(recs):
            n_models[mode] += 1
            g = rec["generation"]
            best = max(rec["attempts"], key=lambda x: x["val_acc"])
            label_ok = 1.0 if g == 0 else rec["pool_label_correct"]
            target = a.strict * label_ok
            cd = rec["circuit"].get("causal_drop_L1", causal.get((mode, lam, seed, g)))
            fail_acc = best["val_acc"] < target
            fail_causal = cd is not None and cd < a.causal
            if not (fail_acc or fail_causal):
                continue
            nxt = recs[i + 1] if i + 1 < len(recs) else None
            w_next = syn_wrong(nxt, lam) if nxt else None
            m = float(np.median(med[(mode, lam)])) if med[(mode, lam)] else None
            rows.append({
                "mode": mode, "real_fraction": lam, "seed": seed, "generation": g,
                "val_acc": round(best["val_acc"], 4), "strict_target": round(target, 4),
                "old_target": round(best["target"], 4),
                "real_acc_all": round(rec["query_acc_all"], 4),
                "induction_best": round(rec["circuit"]["induction_best"], 4),
                "query_ppl_first": round(rec["query_ppl"], 4),
                "causal_drop_L1": None if cd is None else round(cd, 4),
                "fails": "+".join(k for k, v in (("acc", fail_acc), ("causal", fail_causal)) if v),
                "next_pool_syn_wrong_label": None if w_next is None else round(w_next, 4),
                "median_syn_wrong_label_same_lambda": None if m is None else round(m, 4),
                "inflation_x": (None if (w_next is None or not m) else round(w_next / m, 1)),
                "file": os.path.basename(p),
            })
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summary = {m: {"models": n_models[m], "fail_strict": sum(r["mode"] == m for r in rows)}
               for m in n_models}
    json.dump({"summary": summary, "strict": a.strict, "causal": a.causal, "rows": rows},
              open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps(summary))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
