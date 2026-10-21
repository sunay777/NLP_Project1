"""Is the circuit INHERITED across generations, or rebuilt from scratch?

Every generation starts from a fresh random init, so the only channel from
parent to child is the synthetic data. Raw weights live in arbitrary residual
bases, so they are compared through basis-invariant quantities:
  * token-space circuits per head (Elhage et al.-style, final LN ignored):
      QK_h = E W_Q,h^T W_K,h E^T     (which token attends to which)
      OV_h = W_U W_O,h W_V,h E^T     (what an attended token writes)
    compared by Pearson correlation after matching heads within a layer by the
    best permutation (heads have no fixed order across models);
  * linear CKA of the residual stream after each layer on the same real inputs
    (invariant to rotations and isotropic scaling of the residual basis).
The key comparison: parent->child similarity vs the similarity of the child to a
model of the SAME generation from an unrelated lineage (another seed, same
lambda). parent-child > unrelated means something circuit-specific is handed down
through the data; equal means each generation re-derives the same solution.

Run:  python -m experiments.lineage --dir results/v2/main_ext --fractions 0 1 \
          --out results/v2/lineage.json
"""
import argparse
import glob
import itertools
import json
import os
import re

import numpy as np
import torch

from src import data as datamod
from experiments.verify_circuit import load


def token_circuits(model, cfg):
    d, H = cfg.d_model, cfg.n_heads
    dh = d // H
    E = model.tok.weight                                         # (V, d)
    U = model.unembed.weight                                     # (V, d)
    out = []
    for blk in model.blocks:
        W = blk.attn.qkv.weight                                  # (3d, d)
        Wq, Wk, Wv = W[:d], W[d:2 * d], W[2 * d:]
        Wo = blk.attn.proj.weight
        x = blk.ln1(E)                                           # tokens as layer input
        layer = []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            qk = (x @ Wq[sl].T) @ (x @ Wk[sl].T).T               # (V,V)
            ov = (x @ Wv[sl].T) @ Wo[:, sl].T @ U.T              # (V,V)
            layer.append((qk.detach().numpy().ravel(), ov.detach().numpy().ravel()))
        out.append(layer)
    return out


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def matched_similarity(c1, c2):
    """Per layer: mean QK and OV correlation under the best head permutation."""
    res = []
    for l1, l2 in zip(c1, c2):
        H = len(l1)
        best = None
        for perm in itertools.permutations(range(H)):
            qk = np.mean([_corr(l1[h][0], l2[perm[h]][0]) for h in range(H)])
            ov = np.mean([_corr(l1[h][1], l2[perm[h]][1]) for h in range(H)])
            if best is None or qk + ov > best[0] + best[1]:
                best = (qk, ov)
        res.append({"qk": best[0], "ov": best[1]})
    return res


@torch.no_grad()
def residuals(model, idx):
    x = model.tok(idx) + model.pos(torch.arange(idx.shape[1]))[None]
    out = []
    for blk in model.blocks:
        x, _ = blk(x)
        out.append(x.reshape(-1, x.shape[-1]).numpy())
    return out


def cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", default="extended")
    ap.add_argument("--fractions", nargs="+", default=["0", "1"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = {}
    for p in glob.glob(os.path.join(a.dir, "*_gen*.pt")):
        m = re.search(r"rf([0-9.]+)_s(\d+)_rf[0-9.]+_s\d+_gen(\d+)\.pt$", p) or \
            re.search(r"rf([0-9.]+)_s(\d+)_gen(\d+)\.pt$", p)
        if m:
            models[(m.group(1), int(m.group(2)), int(m.group(3)))] = p
    cache = {}

    def get(key):
        if key not in cache:
            cfg, mdl = load(a.mode, models[key])
            idx = datamod.make_batch(cfg, a.n, np.random.default_rng(5), "cpu")["idx"]
            cache[key] = (token_circuits(mdl, cfg), residuals(mdl, idx))
        return cache[key]

    rows = []
    for f in a.fractions:
        seeds = sorted({s for (ff, s, _) in models if ff == f})
        for s in seeds:
            gens = sorted(g for (ff, ss, g) in models if ff == f and ss == s)
            for g in gens[1:]:
                if (f, s, g - 1) not in models:
                    continue
                others = [o for o in seeds if o != s and (f, o, g) in models]
                if not others:
                    continue
                child, parent = get((f, s, g)), get((f, s, g - 1))
                o = others[(seeds.index(s)) % len(others)]          # fixed unrelated partner
                unrel = get((f, o, g))
                ws_par = matched_similarity(child[0], parent[0])
                ws_unr = matched_similarity(child[0], unrel[0])
                ck_par = [cka(x, y) for x, y in zip(child[1], parent[1])]
                ck_unr = [cka(x, y) for x, y in zip(child[1], unrel[1])]
                rows.append({"real_fraction": float(f), "seed": s, "generation": g, "unrelated_seed": o,
                             "parent": {"weights": ws_par, "cka": ck_par},
                             "unrelated": {"weights": ws_unr, "cka": ck_unr}})
                print(f"rf={f} s={s} g={g}: L1 OV parent {ws_par[-1]['ov']:.3f} vs unrelated "
                      f"{ws_unr[-1]['ov']:.3f} | CKA L1 parent {ck_par[-1]:.3f} vs {ck_unr[-1]:.3f}",
                      flush=True)
    summ = {}
    for f in a.fractions:
        rs = [r for r in rows if r["real_fraction"] == float(f)]
        if not rs:
            continue
        L = len(rs[0]["parent"]["cka"])
        d = {}
        for l in range(L):
            for key in ("qk", "ov"):
                diff = [r["parent"]["weights"][l][key] - r["unrelated"]["weights"][l][key] for r in rs]
                d[f"L{l}_{key}_parent_minus_unrelated"] = {
                    "mean": float(np.mean(diff)), "sd": float(np.std(diff)),
                    "frac_positive": float(np.mean(np.array(diff) > 0)), "n": len(diff)}
            diff = [r["parent"]["cka"][l] - r["unrelated"]["cka"][l] for r in rs]
            d[f"L{l}_cka_parent_minus_unrelated"] = {
                "mean": float(np.mean(diff)), "sd": float(np.std(diff)),
                "frac_positive": float(np.mean(np.array(diff) > 0)), "n": len(diff)}
        summ[f] = d
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"rows": rows, "summary": summ}, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
