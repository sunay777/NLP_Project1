"""Independent verification of the induction circuit and extended-mode loss floor.

Loads a trained model (results/single_<mode>/model.pt) and reports, over a large
fresh batch and ALL K query positions (not just the first):
  * per-head previous-token / induction / direct-match (off-by-one) scores
  * accuracy on every supervised position class
  * per-position cross-entropy (to decompose the extended-mode loss floor)
  * zero-ablation of each head (and of whole layers / the L1 induction set)

Run:  python -m experiments.verify_circuit --mode base
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config
from src import data as datamod
from src.model import InductionTransformer


def load(mode, path, **overrides):
    cfg = Config(mode=mode, device="cpu", **overrides)
    m = InductionTransformer(cfg)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return cfg, m


def match_positions(idx, K):
    """For each query j: study position of its symbol (direct match) and label."""
    B = idx.shape[0]
    study_sym = idx[:, 0:2 * K:2]                                  # (B,K)
    q_sym = idx[:, 2 * K:4 * K:2]                                  # (B,K)
    slot = (q_sym[:, :, None] == study_sym[:, None, :]).float().argmax(-1)   # (B,K)
    return 2 * slot, 2 * slot + 1                                   # sym pos, label pos


@torch.no_grad()
def head_table(model, batch, cfg):
    K = cfg.n_pairs
    idx = batch["idx"]
    B = idx.shape[0]
    rows = torch.arange(B)[:, None]
    _, attns = model(idx, return_attn=True)
    sym_pos, lab_pos = match_positions(idx, K)
    qpos = torch.arange(2 * K, 4 * K, 2)[None, :].expand(B, K)
    out = {}
    for l, att in enumerate(attns):
        for h in range(att.shape[1]):
            a = att[:, h]
            prev = torch.stack([a[:, t, t - 1] for t in range(1, 2 * K, 2)]).mean().item()
            prev_all = torch.stack([a[:, t, t - 1] for t in range(1, 2 * K)]).mean().item()
            ind = a[rows, qpos, lab_pos].mean().item()
            ind_first = a[rows[:, 0], qpos[:, 0], lab_pos[:, 0]].mean().item()
            direct = a[rows, qpos, sym_pos].mean().item()
            out[f"L{l}H{h}"] = dict(prev_label=prev, prev_all_study=prev_all,
                                    induction_allq=ind, induction_firstq=ind_first,
                                    direct_match=direct)
    return out


@torch.no_grad()
def position_report(model, batch, cfg):
    K = cfg.n_pairs
    logits = model(batch["idx"])
    tgt = batch["targets"]
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                         ignore_index=-100, reduction="none").reshape(tgt.shape)
    pred = logits.argmax(-1)
    rep = {}
    groups = {"query_labels": list(range(2 * K, 4 * K, 2))}
    if cfg.mode == "extended":
        groups["sq2 (free symbol)"] = [4 * K - 1]
        groups["lq2 (its label)"] = [4 * K]
    for name, ps in groups.items():
        rep[name] = dict(ce=ce[:, ps].mean().item(),
                         acc=(pred[:, ps] == tgt[:, ps]).float().mean().item())
    valid = tgt != -100
    rep["overall_mean_ce"] = ce[valid].mean().item()
    if cfg.mode == "extended":
        # does the model spread mass uniformly over the K in-context symbols?
        p = logits[:, 4 * K - 1].softmax(-1)
        classes = batch["idx"][:, 0:2 * K:2]
        in_ctx = p.gather(1, classes)                              # (B,K)
        rep["sq2_mass_on_context"] = in_ctx.sum(1).mean().item()
        q = (in_ctx / in_ctx.sum(1, keepdim=True)).clamp_min(1e-12)
        H = -(q * q.log()).sum(1)
        rep["sq2_conditional_entropy_over_lnK"] = H.mean().item() / math.log(K)
    rep["theory_floor_lnK_over_Kplus2"] = (math.log(K) / (K + 2)) if cfg.mode == "extended" else 0.0
    return rep


def ablate(model, cfg, heads):
    """Zero the output of the given (layer, head) pairs via a pre-hook on proj."""
    dh = cfg.d_model // cfg.n_heads
    hooks = []
    for l in set(l for l, _ in heads):
        hs = [h for ll, h in heads if ll == l]

        def pre(mod, inp, hs=hs):
            x = inp[0].clone()
            for h in hs:
                x[..., h * dh:(h + 1) * dh] = 0
            return (x,)
        hooks.append(model.blocks[l].attn.proj.register_forward_pre_hook(pre))
    return hooks


@torch.no_grad()
def query_label_acc(model, batch, cfg):
    K = cfg.n_pairs
    ps = list(range(2 * K, 4 * K, 2))
    pred = model(batch["idx"]).argmax(-1)
    return (pred[:, ps] == batch["targets"][:, ps]).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="base", choices=["base", "extended"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n_pairs", type=int, default=4)
    ap.add_argument("--n_symbols", type=int, default=16)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = args.model or f"results/single_{args.mode}/model.pt"
    cfg, model = load(args.mode, path, n_pairs=args.n_pairs, n_symbols=args.n_symbols)
    batch = datamod.make_batch(cfg, 4096, np.random.default_rng(args.seed), "cpu")

    heads = head_table(model, batch, cfg)
    pos = position_report(model, batch, cfg)
    base_acc = query_label_acc(model, batch, cfg)
    abl = {}
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            hk = ablate(model, cfg, [(l, h)])
            abl[f"L{l}H{h}"] = query_label_acc(model, batch, cfg)
            for x in hk: x.remove()
    ind_heads = [(1, h) for h in range(cfg.n_heads)
                 if heads[f"L1H{h}"]["induction_allq"] > 0.2]
    for name, hs in {"all_L0": [(0, h) for h in range(cfg.n_heads)],
                     "all_L1": [(1, h) for h in range(cfg.n_heads)],
                     "L1_induction_set": ind_heads}.items():
        hk = ablate(model, cfg, hs)
        abl[name] = query_label_acc(model, batch, cfg)
        for x in hk: x.remove()

    print(f"== {args.mode}  ({path})  K={cfg.n_pairs}  n_symbols={cfg.n_symbols}")
    print(f"{'head':6s} {'prevLab':>8s} {'prevAll':>8s} {'ind(allq)':>10s} {'ind(q0)':>8s} {'direct':>7s} {'abl_acc':>8s}")
    for k, v in heads.items():
        print(f"{k:6s} {v['prev_label']:8.3f} {v['prev_all_study']:8.3f} {v['induction_allq']:10.3f} "
              f"{v['induction_firstq']:8.3f} {v['direct_match']:7.3f} {abl[k]:8.3f}")
    print(f"query-label acc (all K queries), unablated: {base_acc:.3f}")
    print(f"L1 induction set (score>0.2): {ind_heads}")
    for k in ["all_L0", "all_L1", "L1_induction_set"]:
        print(f"  ablate {k:18s} -> acc {abl[k]:.3f}")
    print("per-position:", json.dumps(pos, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(heads=heads, positions=pos, ablation=abl,
                           unablated_acc=base_acc, induction_set=ind_heads), f, indent=1)


if __name__ == "__main__":
    main()
