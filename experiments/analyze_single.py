"""Milestone 1 interpretability + results figures.

Trains a single generation (base or extended) and saves the artefacts the
write-up needs: training curves, attention-map heatmaps for the identified
previous-token/induction heads, and a symbol-embedding visualisation.

Run:    python -m experiments.analyze_single --mode base
Outputs -> results/single_<mode>/{history.json, training_curves.png,
                                   attention_maps.png, embeddings.png, model.pt}
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import Config, resolve_device
from src import data as datamod
from src import probes
from src.train import train_model


def plot_training_curves(history, out_dir):
    steps = [r["step"] for r in history]
    has_induction = "induction" in history[0]
    fig, axes = plt.subplots(1, 3 if has_induction else 2, figsize=(15, 4))

    axes[0].plot(steps, [r["train_loss"] for r in history])
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("step")

    axes[1].plot(steps, [r["query_acc"] for r in history])
    axes[1].set_title("Query accuracy (val)")
    axes[1].set_xlabel("step")
    axes[1].set_ylim(0, 1.05)

    if has_induction:
        axes[2].plot(steps, [r["induction"] for r in history], color="darkorange")
        axes[2].set_title("Induction-head strength")
        axes[2].set_xlabel("step")
        axes[2].set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def _token_labels(tokens, cfg):
    return [f"S{t}" if t < cfg.n_symbols else f"L{t - cfg.n_symbols}" for t in tokens]


def plot_attention_maps(model, batch, cfg, picked, out_dir):
    model.eval()
    with torch.no_grad():
        _, attns = model(batch["idx"][:1], return_attn=True)
    labels = _token_labels(batch["idx"][0].cpu().numpy(), cfg)

    n_layers = len(attns)
    n_heads = attns[0].shape[1]
    fig, axes = plt.subplots(n_layers, n_heads,
                              figsize=(3.2 * n_heads, 3.2 * n_layers), squeeze=False)
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l][h]
            att = attns[l][0, h].cpu().numpy()
            ax.imshow(att, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"L{l}H{h}", fontsize=9)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=6)

    pt_l, pt_h, pt_s = picked["prev_token_head"]
    ih_l, ih_h, ih_s = picked["induction_head"]
    fig.suptitle(f"Attention maps  |  previous-token head = L{pt_l}H{pt_h} ({pt_s:.2f})"
                 f"   induction head = L{ih_l}H{ih_h} ({ih_s:.2f})")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "attention_maps.png"), dpi=150)
    plt.close(fig)


def plot_embeddings(model, cfg, out_dir):
    W = model.tok.weight.detach().cpu().numpy()
    sym = W[:cfg.n_symbols]
    lab = W[cfg.n_symbols:]

    norm = np.linalg.norm(sym, axis=1, keepdims=True)
    sim = (sym @ sym.T) / (norm @ norm.T + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im = axes[0].imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Symbol embedding cosine similarity")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # 2D PCA over symbols + labels together, so we can see whether the model
    # keeps the two token classes in visibly separate regions of the space.
    W_all = np.concatenate([sym, lab], axis=0)
    W_c = W_all - W_all.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(W_c, full_matrices=False)
    proj = W_c @ Vt[:2].T
    n_sym = sym.shape[0]
    axes[1].scatter(proj[:n_sym, 0], proj[:n_sym, 1], label="symbols", c="tab:blue")
    axes[1].scatter(proj[n_sym:, 0], proj[n_sym:, 1], label="labels", c="tab:red")
    axes[1].set_title("PCA of token embeddings")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "embeddings.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="base", choices=["base", "extended"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(mode=args.mode, steps=args.steps, seed=args.seed)
    out_dir = args.out or f"results/single_{cfg.mode}"
    os.makedirs(out_dir, exist_ok=True)

    device = resolve_device(cfg.device)
    print(f"device={device}  mode={cfg.mode}  vocab={cfg.vocab_size}  seq_len={cfg.max_len}")

    rng = np.random.default_rng(cfg.seed)
    real = lambda bs, r, dev: datamod.make_batch(cfg, bs, r, dev)
    probe_batch = datamod.make_batch(cfg, cfg.eval_batch, rng, device)

    model, history = train_model(cfg, real, val_sampler=real, probe_batch=probe_batch)
    print(f"\nmodel params: {model.num_params():,}")

    scores = probes.head_scores(model, probe_batch, cfg)
    picked = probes.pick_heads(scores)
    pt, ih = picked["prev_token_head"], picked["induction_head"]
    print("\nHead identification:")
    print(f"  previous-token head: layer {pt[0]} head {pt[1]}  score={pt[2]:.3f}")
    print(f"  induction head:      layer {ih[0]} head {ih[1]}  score={ih[2]:.3f}")

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({
            "config": vars(cfg),
            "history": history,
            "picked_heads": {k: list(v) for k, v in picked.items()},
        }, f, indent=2)

    plot_training_curves(history, out_dir)
    plot_attention_maps(model, probe_batch, cfg, picked, out_dir)
    plot_embeddings(model, cfg, out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    print(f"\nsaved artefacts -> {out_dir}/")


if __name__ == "__main__":
    main()
