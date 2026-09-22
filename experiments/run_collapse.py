"""Milestone 2: the Option-A + Option-C experiment.

Sweeps real_fraction across several values and multiple seeds, running a collapse
trajectory for each. Saves all per-generation records to results/collapse.json.

Run (quick):  python -m experiments.run_collapse --mode extended --generations 6 \
                     --fractions 0.0 0.1 0.25 0.5 1.0 --seeds 0 1 2 --steps 1500
"""
import argparse
import json
import os

from src.config import Config
from src.collapse import run_collapse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="extended", choices=["base", "extended"])
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--pool_size", type=int, default=20000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="results/collapse.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_runs = []
    for frac in args.fractions:
        for seed in args.seeds:
            cfg = Config(mode=args.mode, steps=args.steps)
            print(f"\n=== real_fraction={frac}  seed={seed} ===")
            records = run_collapse(cfg, args.generations, frac, seed,
                                   pool_size=args.pool_size,
                                   temperature=args.temperature, verbose=True)
            all_runs.append({"real_fraction": frac, "seed": seed, "records": records})
            with open(args.out, "w") as f:                   # checkpoint after every run
                json.dump({"config": vars(args), "runs": all_runs}, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
