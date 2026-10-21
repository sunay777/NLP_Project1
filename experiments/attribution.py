"""Which training sequences carry the inherited hallucination? (data attribution)

For the model of generation g and the pool it was trained on (saved with
--save_pools), score every training sequence i by its first-order influence on
the model's off-context mass f(theta) on fresh REAL contexts (TracIn at the final
checkpoint, one step of gradient descent on i):

    score_i = - grad_theta f . grad_theta L_i      (> 0: training on i RAISES f)

Efficiency: per-example gradients for 20k sequences x 36k parameters would be
~3 GB. Instead the directional derivative of every example's loss along the
single direction v = grad f is taken with ONE forward-mode JVP per batch
(torch.func.jvp), so all scores cost about one forward pass over the pool.

Reported: enrichment of the top-1% / top-5% most influential sequences for
(a) synthetic sequences with an off-context sq2 (the parent's hallucinations),
(b) synthetic sequences with a wrong query label, (c) synthetic sequences in
general, each relative to its base rate in the pool. The causal counterpart is
the scrub experiment (jobs_v2: scrub_offctx), which removes (a) from every pool.
Chen et al. (2026) trace interpretable units to training data with influence
methods; this is the same idea at the level of a behaviour (off-context mass).

Run:  python -m experiments.attribution --dir results/v2/main_ext --fraction 0 \
          --seeds 0 1 2 --gens 2 5 10 --out results/v2/attribution.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

from src import data as datamod
from experiments.verify_circuit import load


def offctx_mass(logits, idx, K):
    p = torch.softmax(logits[:, 4 * K - 1], -1)
    return 1 - p.gather(1, idx[:, 0:2 * K:2]).sum(1).mean()


def per_example_loss(logits, targets):
    V = logits.shape[-1]
    l = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100,
                        reduction="none").reshape(targets.shape)
    n = (targets != -100).sum(1).clamp_min(1)
    return l.sum(1) / n


def scores(model, cfg, pool, n_test=2048, bs=2048):
    K = cfg.n_pairs
    params = {k: v.detach() for k, v in model.named_parameters()}
    test = datamod.make_batch(cfg, n_test, np.random.default_rng(31337), "cpu")
    f = lambda p: offctx_mass(functional_call(model, p, (test["idx"],)), test["idx"], K)
    p_req = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    gf = torch.autograd.grad(f(p_req), list(p_req.values()))
    v = {k: g for k, g in zip(p_req, gf)}
    idx = pool["idx"].long()
    tgt = pool["targets"].long()
    out = []
    for i in range(0, idx.shape[0], bs):
        bi, bt = idx[i:i + bs], tgt[i:i + bs]
        fn = lambda p: per_example_loss(functional_call(model, p, (bi,)), bt)
        _, dl = jvp(fn, (params,), (v,))                       # (B,) = grad L_i . grad f
        out.append(-dl)
    return torch.cat(out).numpy(), float(f(params))


def enrichment(score, flags, qs=(0.01, 0.05)):
    order = np.argsort(-score)
    res = {}
    for name, fl in flags.items():
        base = float(fl.mean())
        res[name] = {"base_rate": base}
        for q in qs:
            top = order[:max(1, int(q * len(score)))]
            r = float(fl[top].mean())
            res[name][f"top{int(q * 100)}pct_rate"] = r
            res[name][f"top{int(q * 100)}pct_enrichment"] = (r / base) if base > 0 else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", default="extended")
    ap.add_argument("--fraction", default="0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gens", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        for g in a.gens:
            mp = os.path.join(a.dir, f"rf{a.fraction}_s{s}_rf{a.fraction}_s{s}_gen{g}.pt")
            pp = os.path.join(a.dir, f"rf{a.fraction}_s{s}_rf{a.fraction}_s{s}_pool{g}.pt")
            if not (os.path.exists(mp) and os.path.exists(pp)):
                print(f"skip s={s} g={g}: missing {mp if not os.path.exists(mp) else pp}")
                continue
            cfg, model = load(a.mode, mp)
            pool = torch.load(pp)
            K, S = cfg.n_pairs, cfg.n_symbols
            idx = pool["idx"].long()
            syn = pool["is_syn"].numpy()
            off = ~(idx[:, 0:2 * K:2] == idx[:, 4 * K][:, None]).any(1).numpy()
            qp = list(range(2 * K, 4 * K, 2))
            wrong = (pool["targets"][:, qp] != pool["true_targets"][:, qp]).any(1).numpy()
            sc, fval = scores(model, cfg, pool)
            flags = {"synthetic_offctx_sq2": syn & off, "synthetic_wrong_label": syn & wrong,
                     "synthetic": syn.astype(bool)}
            enr = enrichment(sc, flags)
            rows.append({"seed": s, "generation": g, "offctx_mass": fval, "n_pool": int(len(sc)),
                         "enrichment": enr,
                         "mean_score_offctx_rows": float(sc[syn & off].mean()) if (syn & off).any() else None,
                         "mean_score_other_rows": float(sc[~(syn & off)].mean())})
            e = enr["synthetic_offctx_sq2"]
            print(f"s={s} g={g}: off-ctx rows base {e['base_rate']:.4f} -> top1% {e['top1pct_rate']:.3f} "
                  f"({e['top1pct_enrichment']:.1f}x), top5% {e['top5pct_enrichment']:.1f}x", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
