"""Where does the hallucinated (off-context) mass come from, and where does the
query-NLL erosion live? Compares saved gen-0 vs final-generation models.

1. Direct logit attribution (DLA) at the sq2 position (4K-1). With the final
   LayerNorm scale frozen at its value on the full residual, the logits split
   exactly into components: embeddings (token + position), every attention head,
   and the LN bias. For each component we report its mean contribution to the
   off-context-minus-in-context logit gap (> 0 pushes mass off-context).
2. Which off-context symbols receive the mass? Per context, Spearman correlation
   of p(s) over the non-context symbols with s's max embedding cosine to the
   in-context symbols (fuzzy copy => positive), plus the share of variance in
   p(s) explained by symbol identity alone (a learned, context-free prior => high).
3. Query slots, last layer: attention that leaks to NON-matching study labels
   (QK error) vs the OV label-copy score (OV error), and the query NLL.
4. Causal (not just direct) effects: mean-ablate each head in turn and record the
   change in off-context mass and in query NLL. A head whose ablation REDUCES
   off-context mass is writing the hallucination (directly or via later layers).

Run:  python -m experiments.mechanism --glob "results/sweep/extended_all*_rf{f}_s*_gen{g}.pt" \
          --fractions 0 1 --gens 0 5 --out results/v2/mechanism.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from src import data as datamod
from src.probes import ov_copy_scores, match_positions
from experiments.verify_circuit import load


def _rank(x):
    return np.argsort(np.argsort(x, axis=-1), axis=-1).astype(float)


def spearman_rows(a, b):
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    ok = den > 0
    return float(((ra * rb).sum(1)[ok] / den[ok]).mean())


@torch.no_grad()
def components(model, idx, cfg):
    """Residual at every position split into embed / per-head pieces + LN constants."""
    B, T = idx.shape
    d, H = cfg.d_model, cfg.n_heads
    dh = d // H
    x = model.tok(idx) + model.pos(torch.arange(T))[None]
    comps = {"embed": x.clone()}
    for l, blk in enumerate(model.blocks):
        h_in = blk.ln1(x)
        attn = blk.attn
        qkv = attn.qkv(h_in).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if attn.use_rope:
            from src.model import _rope_cache, _apply_rope
            cos, sin = _rope_cache(T, dh, idx.device)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / dh ** 0.5
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        z = att @ v                                                      # (B,H,T,dh)
        Wo = attn.proj.weight                                            # (d, d)
        tot = torch.zeros_like(x)
        for h in range(H):
            o = z[:, h] @ Wo[:, h * dh:(h + 1) * dh].T                   # (B,T,d)
            comps[f"L{l}H{h}"] = o
            tot = tot + o
        x = x + tot
        if blk.use_mlp:
            m = blk.mlp(blk.ln2(x))
            comps[f"L{l}mlp"] = m
            x = x + m
    return comps, x


@torch.no_grad()
def dla_sq2(model, batch, cfg):
    K, S = cfg.n_pairs, cfg.n_symbols
    idx = batch["idx"]
    pos = 4 * K - 1
    comps, x = components(model, idx, cfg)
    xf = x[:, pos]
    mu, sd = xf.mean(-1, keepdim=True), xf.std(-1, unbiased=False, keepdim=True)
    sd = (sd ** 2 + model.lnf.eps).sqrt()
    W = model.unembed.weight                                              # (V,d)
    g = model.lnf.weight
    in_ctx = torch.zeros(idx.shape[0], cfg.vocab_size, dtype=torch.bool)
    in_ctx.scatter_(1, idx[:, 0:2 * K:2], True)
    off = ~in_ctx
    off[:, S:] = False                                                    # symbols only
    out = {}
    for name, c in comps.items():
        cc = c[:, pos]
        lg = (((cc - cc.mean(-1, keepdim=True)) / sd) * g) @ W.T           # (B,V)
        gap = (lg * off).sum(1) / off.sum(1) - (lg * in_ctx).sum(1) / in_ctx.sum(1)
        out[name] = float(gap.mean())
    # sanity: components + bias reproduce the model's logits
    full = model(idx)[:, pos]
    recon = sum((((c[:, pos] - c[:, pos].mean(-1, keepdim=True)) / sd) * g) @ W.T
                for c in comps.values()) + model.lnf.bias @ W.T
    out["_recon_max_abs_err"] = float((full - recon).abs().max())
    return out


@torch.no_grad()
def offctx_targets(model, batch, cfg):
    K, S = cfg.n_pairs, cfg.n_symbols
    idx = batch["idx"]
    p = torch.softmax(model(idx)[:, 4 * K - 1], -1)[:, :S].numpy()        # (B,S)
    ctx = idx[:, 0:2 * K:2].numpy()
    E = model.tok.weight[:S]
    E = E / E.norm(dim=1, keepdim=True)
    cos = (E @ E.T).numpy()
    B = idx.shape[0]
    P, C, SYM = [], [], []
    for b in range(B):
        mask = np.ones(S, bool)
        mask[ctx[b]] = False
        P.append(p[b, mask])
        C.append(cos[np.ix_(np.where(mask)[0], ctx[b])].max(1))
        SYM.append(np.where(mask)[0])
    P, C, SYM = np.array(P), np.array(C), np.array(SYM)
    pn = P / P.sum(1, keepdims=True)
    # share of variance of the normalised off-context distribution explained by symbol id
    flat_p, flat_s = pn.ravel(), SYM.ravel()
    sym_mean = np.array([flat_p[flat_s == s].mean() for s in range(S)])
    ss_tot = ((flat_p - flat_p.mean()) ** 2).sum()
    ss_sym = ((sym_mean[flat_s] - flat_p.mean()) ** 2).sum()
    return {"off_context_mass": float(P.sum(1).mean()),
            "spearman_p_vs_embed_cos": spearman_rows(P, C),
            "symbol_identity_r2": float(ss_sym / ss_tot) if ss_tot > 0 else None,
            "top_symbol_share": float(np.sort(sym_mean)[-3:].sum() / sym_mean.sum())}


@torch.no_grad()
def query_qk_ov(model, batch, cfg):
    K = cfg.n_pairs
    idx = batch["idx"]
    B = idx.shape[0]
    rows = torch.arange(B)[:, None]
    _, attns = model(idx, return_attn=True)
    a = attns[-1]                                                         # (B,H,T,T)
    _, lab_pos = match_positions(idx, K)
    qpos = torch.arange(2 * K, 4 * K, 2)[None].expand(B, K)
    all_lab = torch.arange(1, 2 * K, 2)
    out = []
    ov = ov_copy_scores(model, cfg)[-1]
    for h in range(a.shape[1]):
        ah = a[:, h]
        to_match = ah[rows, qpos, lab_pos]                                # (B,K)
        to_labels = ah[:, :, all_lab][rows, qpos].sum(-1)                 # (B,K) all study labels
        out.append({"head": h, "attn_match_label": float(to_match.mean()),
                    "attn_nonmatch_labels": float((to_labels - to_match).mean()),
                    "ov_copy": ov[h]})
    logp = torch.log_softmax(model(idx)[:, 2 * K:4 * K:2], -1)
    tgt = idx[:, 2 * K + 1:4 * K:2]
    nll = float(-logp.gather(-1, tgt[..., None]).mean())
    best = max(out, key=lambda r: r["attn_match_label"])
    return {"per_head": out, "query_nll": nll,
            "best_head_leak": best["attn_nonmatch_labels"], "best_head_ov": best["ov_copy"]}


@torch.no_grad()
def ablation_effects(model, batch, cfg):
    from src.probes import ablate_heads, head_means
    K = cfg.n_pairs
    idx = batch["idx"]
    tgt = idx[:, 2 * K + 1:4 * K:2]

    def measure():
        lg = model(idx)
        p = torch.softmax(lg[:, 4 * K - 1], -1)
        off = 1 - p.gather(1, idx[:, 0:2 * K:2]).sum(1).mean().item()
        lp = torch.log_softmax(lg[:, 2 * K:4 * K:2], -1)
        return off, float(-lp.gather(-1, tgt[..., None]).mean())
    off0, nll0 = measure()
    means = head_means(model, batch)
    out = {}
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            hk = ablate_heads(model, cfg, [(l, h)], means)
            off, nll = measure()
            for x in hk:
                x.remove()
            out[f"L{l}H{h}"] = {"d_offctx": off - off0, "d_query_nll": nll - nll0}
    return {"base_offctx": off0, "base_query_nll": nll0, "per_head": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="pattern with {f} and {g} placeholders")
    ap.add_argument("--mode", default="extended")
    ap.add_argument("--fractions", nargs="+", default=["0", "1"])
    ap.add_argument("--gens", nargs="+", default=["0", "5"])
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"rows": [], "glob": a.glob}
    for f in a.fractions:
        for g in a.gens:
            for p in sorted(glob.glob(a.glob.format(f=f, g=g))):
                s = int(re.search(r"_s(\d+)_gen", p).group(1))
                cfg, m = load(a.mode, p)
                batch = datamod.make_batch(cfg, a.n, np.random.default_rng(1234), "cpu")
                row = {"real_fraction": float(f), "generation": int(g), "seed": s, "path": p,
                       "dla_offctx_gap": dla_sq2(m, batch, cfg),
                       "offctx": offctx_targets(m, batch, cfg),
                       "query": query_qk_ov(m, batch, cfg),
                       "ablation": ablation_effects(m, batch, cfg)}
                res["rows"].append(row)
                d = row["dla_offctx_gap"]
                top = sorted(((k, v) for k, v in d.items() if not k.startswith("_")),
                             key=lambda kv: -abs(kv[1]))[:3]
                print(f"rf={f} gen={g} s={s} off={row['offctx']['off_context_mass']:.4f} "
                      f"rho(cos)={row['offctx']['spearman_p_vs_embed_cos']:.2f} "
                      f"symR2={row['offctx']['symbol_identity_r2']:.2f} "
                      f"leak={row['query']['best_head_leak']:.4f} ov={row['query']['best_head_ov']:.1f} "
                      f"nll={row['query']['query_nll']:.5f} | DLA top {top} "
                      f"(recon err {d['_recon_max_abs_err']:.1e})")
    # summary: mean over seeds per (fraction, generation)
    summ = {}
    for r in res["rows"]:
        k = f"rf{r['real_fraction']:g}_gen{r['generation']}"
        summ.setdefault(k, []).append(r)
    res["summary"] = {}
    for k, rs in summ.items():
        names = [n for n in rs[0]["dla_offctx_gap"] if not n.startswith("_")]
        res["summary"][k] = {
            "n_seeds": len(rs),
            "dla_offctx_gap_mean": {n: float(np.mean([r["dla_offctx_gap"][n] for r in rs])) for n in names},
            **{f"offctx_{q}": float(np.mean([r["offctx"][q] for r in rs]))
               for q in ("off_context_mass", "spearman_p_vs_embed_cos", "symbol_identity_r2",
                         "top_symbol_share")},
            **{f"query_{q}": float(np.mean([r["query"][q] for r in rs]))
               for q in ("query_nll", "best_head_leak", "best_head_ov")},
            # smallest relative change in off-context mass over single-head
            # mean-ablations: < 0 would mean some head WRITES the hallucination;
            # > 0 means every head suppresses it (ablating any head adds mass)
            "min_rel_d_offctx_single_head": float(np.mean([
                min(v["d_offctx"] for v in r["ablation"]["per_head"].values())
                / r["ablation"]["base_offctx"] for r in rs])),
            "n_heads_increasing_offctx_when_ablated": float(np.mean([
                sum(v["d_offctx"] > 0 for v in r["ablation"]["per_head"].values()) for r in rs]))}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in v.items() if kk != "dla_offctx_gap_mean"}
                      for k, v in res["summary"].items()}, indent=1))
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
