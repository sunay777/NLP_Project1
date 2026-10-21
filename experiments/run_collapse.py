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
    ap.add_argument("--gen0_pool", action="store_true",
                    help="train gen 0 on a fixed real pool of --pool_size (removes the "
                         "unlimited->finite data confound between gen 0 and gen 1)")
    ap.add_argument("--save_models", nargs="?", const="ends", default="none",
                    choices=["none", "ends", "all"],
                    help="save weights next to --out: 'ends' = gen 0 + final (bare flag), "
                         "'all' = every generation")
    # ---- v2 options (defaults reproduce the original sweep) ----
    ap.add_argument("--device", default="auto", help="cpu | mps | cuda | auto")
    ap.add_argument("--real_gate", type=float, default=None,
                    help="gen-0 accuracy gate (default: 0.9, or --gate if --gate_causal set)")
    ap.add_argument("--gate_causal", type=float, default=None,
                    help="strict gate: also require causal_drop_L1 >= this on real data")
    ap.add_argument("--save_pools", action="store_true", help="save every training pool")
    ap.add_argument("--pulse_gen", type=int, default=None)
    ap.add_argument("--pulse_label_noise", type=float, default=0.0)
    ap.add_argument("--pulse_offcontext", type=float, default=0.0)
    ap.add_argument("--accumulate", action="store_true",
                    help="train on the union of all previous pools (needs --gen0_pool)")
    ap.add_argument("--full_lm", action="store_true", help="supervise every next token")
    ap.add_argument("--gen_context", action="store_true",
                    help="fully synthetic sequences: model writes the context too (needs --full_lm)")
    ap.add_argument("--scrub", default="none", choices=["none", "offctx", "labels", "both"])
    ap.add_argument("--sq2_zipf", type=float, default=0.0)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--use_mlp", action="store_true")
    ap.add_argument("--n_pairs", type=int, default=4)
    ap.add_argument("--n_symbols", type=int, default=16)
    args = ap.parse_args()
    real_gate = args.real_gate if args.real_gate is not None else (
        args.gate if args.gate_causal is not None else 0.9)

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
            cfg = Config(mode=args.mode, steps=args.steps, eval_every=args.eval_every,
                         device=args.device, sq2_zipf=args.sq2_zipf, full_lm=args.full_lm,
                         n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads,
                         use_mlp=args.use_mlp, n_pairs=args.n_pairs, n_symbols=args.n_symbols)
            print(f"\n=== {args.mode} gen_queries={args.gen_queries} T={args.temperature} "
                  f"real_fraction={frac}  seed={seed} ===", flush=True)
            records = run_collapse(cfg, args.generations, frac, seed,
                                   pool_size=args.pool_size,
                                   temperature=args.temperature,
                                   gen_queries=args.gen_queries,
                                   gate=args.gate, real_gate=real_gate,
                                   max_retries=args.max_retries,
                                   verbose=True, gen0_pool=args.gen0_pool,
                                   gate_causal=args.gate_causal,
                                   save_models=args.save_models, save_pools=args.save_pools,
                                   pulse_gen=args.pulse_gen,
                                   pulse_label_noise=args.pulse_label_noise,
                                   pulse_offcontext=args.pulse_offcontext,
                                   accumulate=args.accumulate, gen_context=args.gen_context,
                                   scrub=args.scrub,
                                   save_prefix=f"{os.path.splitext(args.out)[0]}_rf{frac:g}_s{seed}")
            all_runs.append({"real_fraction": frac, "seed": seed, "records": records})
            with open(args.out, "w") as f:
                json.dump({"config": vars(args), "runs": all_runs}, f, indent=1)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
