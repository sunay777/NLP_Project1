"""Milestone 2: the Option-A + Option-C experiment.

Sweeps real_fraction across several values and multiple seeds, running a collapse
trajectory for each. Saves all per-generation records to --out (JSON,
checkpointed after every run; re-running with the same --out resumes and skips
(real_fraction, seed) pairs already present).

Run:  python -m experiments.run_collapse --mode extended --generations 6 \
          --fractions 0.0 0.1 0.25 0.5 1.0 --seeds 0 1 2 --steps 6000 \
          --gen_queries all --out results/collapse_extended_all.json
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
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--pool_size", type=int, default=20000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gen_queries", default="first", choices=["first", "all"])
    ap.add_argument("--gate", type=float, default=0.9)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--out", default="results/collapse.json")
    ap.add_argument("--save_models", action="store_true",
                    help="save gen-0 and final-gen weights next to --out")
    args = ap.parse_args()

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    all_runs = []
    if os.path.exists(args.out):
        all_runs = json.load(open(args.out)).get("runs", [])
    done = {(r["real_fraction"], r["seed"]) for r in all_runs}

    for frac in args.fractions:
        for seed in args.seeds:
            if (frac, seed) in done:
                print(f"skip real_fraction={frac} seed={seed} (already in {args.out})")
                continue
            cfg = Config(mode=args.mode, steps=args.steps, eval_every=args.eval_every)
            print(f"\n=== {args.mode} gen_queries={args.gen_queries} T={args.temperature} "
                  f"real_fraction={frac}  seed={seed} ===", flush=True)
            records = run_collapse(cfg, args.generations, frac, seed,
                                   pool_size=args.pool_size,
                                   temperature=args.temperature,
                                   gen_queries=args.gen_queries,
                                   gate=args.gate, max_retries=args.max_retries,
                                   verbose=True,
                                   save_prefix=(f"{os.path.splitext(args.out)[0]}_rf{frac:g}_s{seed}"
                                                if args.save_models else None))
            all_runs.append({"real_fraction": frac, "seed": seed, "records": records})
            with open(args.out, "w") as f:
                json.dump({"config": vars(args), "runs": all_runs}, f, indent=1)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
