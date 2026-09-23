"""Re-probe the saved gen-0 and final-generation models of a sweep with the
current (variant-aware) circuit probe. Output: one row per saved model.

Run:  python -m experiments.reprobe_saved --glob "results/sweep/extended_all*_gen*.pt" \
          --mode extended --out results/figures/reprobe_extended.csv
"""
import argparse
import csv
import glob
import re

import numpy as np

from src import data as datamod
from src.probes import circuit_probe
from experiments.verify_circuit import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(a.glob)):
        mt = re.search(r"_rf([0-9.]+)_s(\d+)_gen(\d+)\.pt$", p)
        if not mt:
            continue
        cfg, m = load(a.mode, p)
        b = datamod.make_batch(cfg, 4096, np.random.default_rng(77), "cpu")
        c = circuit_probe(m, b, cfg)
        rows.append([float(mt.group(1)), int(mt.group(2)), int(mt.group(3)), c["circuit_type"],
                     round(c["query_acc_all"], 4), round(c["induction_best"], 4),
                     round(c["induction_legacy"], 4), round(c["causal_drop"], 4),
                     round(c["causal_drop_L1"], 4),
                     round(min(h["ov_copy"] for h in c["per_head"]), 2),
                     round(max(h["ov_copy"] for h in c["per_head"]), 2)])
    rows.sort()
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["real_fraction", "seed", "generation", "circuit_type", "acc_all",
                    "induction_best", "induction_legacy", "causal_drop_set",
                    "causal_drop_L1", "min_ov_copy_L1", "max_ov_copy_L1"])
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
