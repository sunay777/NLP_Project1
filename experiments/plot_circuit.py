"""Figure 1: the induction circuit in a trained model (paper-sized).

  (a) batch-averaged layer-0 attention along the three routes that feed the two
      circuit variants: label->its symbol (t-1 at study labels; canonical) and,
      at study symbols, ->previous label (t-1) and ->previous symbol (t-2)
      (together these build the shifted variant's key/value);
  (b) batch-averaged attention of the best L1 induction head from each query to
      positions relative to its match (match_sym-1 ... match_sym+3);
  (c) query accuracy after zero-ablating each head and each whole layer
      (dashed line = unablated; variant-agnostic).
Averaging over a large batch (not one example) shows the typical pattern.

Run:  python -m experiments.plot_circuit --mode base --model results/single_base/model.pt \
          --out results/figures/fig_circuit_base.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import data as datamod
from src.probes import (ablate_heads, circuit_probe, match_positions,
                        query_label_accuracy)
from experiments.verify_circuit import load
from experiments.plot_collapse import style, SERIES, INK2

OFFSETS = [-1, 0, 1, 2, 3]
OFF_LABELS = ["sym−1", "match\nsym", "match\nlabel", "sym+2\n(shifted)", "sym+3"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="base")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    style()
    cfg, m = load(a.mode, a.model)
    K = cfg.n_pairs
    b = datamod.make_batch(cfg, 4096, np.random.default_rng(11), "cpu")
    _, attns = m(b["idx"], return_attn=True)
    circ = circuit_probe(m, b, cfg)
    B = b["idx"].shape[0]
    rows = torch.arange(B)[:, None]

    # (a) L0 routes
    lab_t = torch.arange(1, 2 * K, 2)                          # study labels
    sym_t = torch.arange(2, 2 * K, 2)                          # study symbols (not the first)
    l0 = []
    for h in range(cfg.n_heads):
        A = attns[0][:, h]
        l0.append([A[:, lab_t, lab_t - 1].mean().item(),
                   A[:, sym_t, sym_t - 1].mean().item(),
                   A[:, sym_t, sym_t - 2].mean().item()])
    l0 = np.array(l0)                                          # (H, 3)

    # (b) L1: at query positions, mass at offsets from the matching study symbol
    sym_pos, _ = match_positions(b["idx"], K)
    qpos = torch.arange(2 * K, 4 * K, 2)[None, :].expand(B, K)
    l1 = []
    for h in range(cfg.n_heads):
        v = []
        for off in OFFSETS:
            p = sym_pos + off
            valid = (p >= 0) & (p < qpos)
            v.append((attns[-1][:, h][rows, qpos, p.clamp(0, cfg.max_len - 1)] * valid).sum().item()
                     / max(1, valid.sum().item()))
        l1.append(v)
    l1 = np.array(l1)                                          # (H, 5)

    # (c) ablations
    base_acc = query_label_accuracy(m, b, cfg)
    names, accs = [], []
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            hk = ablate_heads(m, cfg, [(l, h)])
            names.append(f"L{l}H{h}"); accs.append(query_label_accuracy(m, b, cfg))
            for x in hk: x.remove()
    for nm, hs in [("all\nL0", [(0, h) for h in range(cfg.n_heads)]),
                   ("all\nL1", [(cfg.n_layers - 1, h) for h in range(cfg.n_heads)])]:
        hk = ablate_heads(m, cfg, hs)
        names.append(nm); accs.append(query_label_accuracy(m, b, cfg))
        for x in hk: x.remove()

    fig = plt.figure(figsize=(11, 3.2))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.25, 0.05, 0.3, 1.9], wspace=0.35)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), None,
          fig.add_subplot(gs[0, 4])]
    cax = fig.add_subplot(gs[0, 2])
    for i, (M, xl, ttl, xt) in enumerate([
            (l0, "query position → attended position", "(a) Layer 0 heads",
             ["label→\nits sym\n(t−1)", "sym→\nprev label\n(t−1)", "sym→\nprev sym\n(t−2)"]),
            (l1, "attended position (relative to match)", "(b) Layer 1 heads",
             OFF_LABELS)]):
        im = ax[i].imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax[i].set_xticks(range(M.shape[1])); ax[i].set_xticklabels(xt, fontsize=7)
        ax[i].set_yticks(range(cfg.n_heads))
        ax[i].set_yticklabels([f"L{i}H{h}" for h in range(cfg.n_heads)])
        ax[i].set_title(ttl); ax[i].set_xlabel(xl, fontsize=8); ax[i].grid(False)
        for (r, c), v in np.ndenumerate(M):
            if v >= 0.1:
                ax[i].text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                           color="white" if v > 0.55 else "#0b0b0b")
    fig.colorbar(im, cax=cax, label="mean attention")

    x = np.arange(len(names))
    cols = [SERIES["induction"] if ("L1" in n) else SERIES["diversity"] for n in names]
    ax[3].bar(x, accs, color=cols, width=0.7)
    ax[3].axhline(base_acc, ls="--", color=INK2, lw=1)
    ax[3].axhline(1 / cfg.n_labels, ls=":", color=INK2, lw=1)
    ax[3].text(-0.45, 1 / cfg.n_labels + 0.02, "chance", fontsize=7,
               color=INK2, ha="left")
    ax[3].set_xticks(x); ax[3].set_xticklabels(names, fontsize=7, rotation=0)
    ax[3].set_ylim(0, 1.05); ax[3].set_ylabel("query accuracy")
    ax[3].set_title("(c) Zero-ablation")
    from matplotlib.patches import Patch
    ax[3].legend(handles=[Patch(color=SERIES["diversity"], label="layer 0"),
                          Patch(color=SERIES["induction"], label="layer 1")],
                 loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
    for xi, v in zip(x, accs):
        ax[3].text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5, color="#0b0b0b")
    if a.title:
        fig.suptitle(a.title, fontsize=10, y=1.04)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=220, bbox_inches="tight")
    print(f"saved {a.out} | circuit_type={circ['circuit_type']} induction_best="
          f"{circ['induction_best']:.3f} causal_drop={circ['causal_drop']:.3f}")


if __name__ == "__main__":
    main()
