"""Train one extended/base model with config overrides and report when (if) the
induction transition happens and the per-position loss decomposition.

Used to verify (i) the extended-mode loss floor ln(K)/(K+2) across K, and
(ii) whether a given --steps budget reliably crosses the phase transition.

Run:  python -m experiments.check_extended --mode extended --steps 6000 --seed 1 --n_pairs 4
"""
import argparse
import json
import os

import numpy as np
import torch

from src.config import Config
from src import data as datamod
from src.train import train_model
from experiments.verify_circuit import position_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="extended")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_pairs", type=int, default=4)
    ap.add_argument("--n_symbols", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = Config(mode=a.mode, steps=a.steps, seed=a.seed, n_pairs=a.n_pairs,
                 n_symbols=a.n_symbols, eval_every=100)
    rng = np.random.default_rng(10_000 + a.seed)
    real = lambda bs, r, dev: datamod.make_batch(cfg, bs, r, dev)
    probe = datamod.make_batch(cfg, cfg.eval_batch, rng, "cpu")
    model, hist = train_model(cfg, real, val_sampler=real, probe_batch=probe, log=False)
    trans = next((r["step"] for r in hist if r["query_acc"] > 0.95), None)
    ev = datamod.make_batch(cfg, 8192, np.random.default_rng(999), "cpu")
    pos = position_report(model, ev, cfg)
    res = dict(args=vars(a), transition_step=trans, final=hist[-1], positions=pos,
               history=hist)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    torch.save(model.state_dict(), a.out.replace(".json", ".pt"))
    print(json.dumps(dict(args=vars(a), transition_step=trans,
                          final_acc=hist[-1]["query_acc"], final_ind=hist[-1]["induction"],
                          eval_ce=pos["overall_mean_ce"],
                          floor=pos["theory_floor_lnK_over_Kplus2"])))


if __name__ == "__main__":
    main()
