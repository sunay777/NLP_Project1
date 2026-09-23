"""Hyper-parameter selection on a VALIDATION split (brief requirement).

Grid-searches gen-0 training on real data and selects by validation loss; the
selected config is then evaluated once on a disjoint TEST split. Three disjoint
random streams keep train / validation / test data separate.

Selection criterion = mean validation cross-entropy over all supervised positions
(for extended mode this has the irreducible floor ln K/(K+2), identical across
configs, so ranking is unaffected). Ties / near-ties are broken by the fraction
of seeds that pass the induction transition, then by earlier transition.

Run:  python -m experiments.tune_hparams --mode base --steps 4000 --seeds 0 1 \
          --out results/tuning_base.json
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config
from src import data as datamod
from src.train import train_model
from src.probes import circuit_probe


@torch.no_grad()
def eval_loss_acc(model, batch):
    model.eval()
    logits = model(batch["idx"])
    V = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, V), batch["targets"].reshape(-1),
                           ignore_index=-100).item()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="base")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--lr", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--warmup", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--d_model", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    results = []
    if os.path.exists(a.out):
        results = json.load(open(a.out))["results"]
    done = {(r["lr"], r["warmup"], r["d_model"], r["seed"]) for r in results}

    for lr, wu, dm, seed in itertools.product(a.lr, a.warmup, a.d_model, a.seeds):
        if (lr, wu, dm, seed) in done:
            continue
        cfg = Config(mode=a.mode, steps=a.steps, lr=lr, warmup=wu, d_model=dm,
                     seed=seed, device="cpu", eval_every=250)
        val = datamod.make_batch(cfg, 4096, np.random.default_rng(1_000 + seed), "cpu")
        real = lambda bs, r, dev: datamod.make_batch(cfg, bs, r, dev)
        t0 = time.time()
        model, hist = train_model(cfg, real, val_sampler=real, probe_batch=val, log=False)
        circ = circuit_probe(model, val, cfg)
        rec = dict(lr=lr, warmup=wu, d_model=dm, seed=seed,
                   val_loss=eval_loss_acc(model, val),
                   val_acc_all=circ["query_acc_all"],
                   induction_best=circ["induction_best"],
                   causal_drop=circ["causal_drop"],
                   transitioned=circ["query_acc_all"] > 0.95,
                   transition_step=next((r["step"] for r in hist if r["query_acc"] > 0.95), None),
                   seconds=round(time.time() - t0, 1))
        results.append(rec)
        print(json.dumps(rec), flush=True)
        json.dump({"args": vars(a), "results": results}, open(a.out, "w"), indent=1)

    # aggregate per config
    agg = {}
    for r in results:
        k = (r["lr"], r["warmup"], r["d_model"])
        agg.setdefault(k, []).append(r)
    table = []
    for k, rs in agg.items():
        ts = [r["transition_step"] for r in rs if r["transition_step"] is not None]
        table.append(dict(lr=k[0], warmup=k[1], d_model=k[2],
                          val_loss=float(np.mean([r["val_loss"] for r in rs])),
                          pass_rate=float(np.mean([r["transitioned"] for r in rs])),
                          mean_transition=float(np.mean(ts)) if ts else None,
                          n=len(rs)))
    table.sort(key=lambda t: (round(t["val_loss"], 3), -t["pass_rate"],
                              t["mean_transition"] if t["mean_transition"] is not None else 1e9))
    best = table[0]
    print("\nconfig ranking (by validation loss):")
    for t in table:
        print(f"  lr={t['lr']:<7g} warmup={t['warmup']:<4d} d_model={t['d_model']:<3d} "
              f"val_loss={t['val_loss']:.4f} pass={t['pass_rate']:.2f} "
              f"transition~{t['mean_transition']}")

    # held-out TEST evaluation of the selected config (fresh seed, disjoint stream)
    cfg = Config(mode=a.mode, steps=a.steps, lr=best["lr"], warmup=best["warmup"],
                 d_model=best["d_model"], seed=99, device="cpu")
    test = datamod.make_batch(cfg, 8192, np.random.default_rng(2_000_000), "cpu")
    real = lambda bs, r, dev: datamod.make_batch(cfg, bs, r, dev)
    model, _ = train_model(cfg, real, log=False)
    circ = circuit_probe(model, test, cfg)
    test_rec = dict(selected=best, test_loss=eval_loss_acc(model, test),
                    test_acc_all=circ["query_acc_all"],
                    induction_best=circ["induction_best"])
    print("\nselected + test:", json.dumps(test_rec))
    json.dump({"args": vars(a), "results": results, "ranking": table, "test": test_rec},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
